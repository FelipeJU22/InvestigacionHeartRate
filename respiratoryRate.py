import cv2
import numpy as np
import peakutils
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
from datetime import datetime
import time

# Configuración de frecuencia de respiración (en Hz)
FREQ_MIN = 0.1  # 6 respiraciones por minuto
FREQ_MAX = 0.55  # 30 respiraciones por minuto


# Filtro Butterworth Paso Banda
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


# Extraer señal de movimiento basada en cambios de luminancia
def extract_motion_signal(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or not cap.isOpened():
        print("Error: No se pudo leer el video o FPS inválido.")
        cap.release()
        return None, None

    motion_signal = []
    prev_frame = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            diff = np.mean(np.abs(gray.astype(np.float32) - prev_frame))
            motion_signal.append(diff)
        prev_frame = gray
        frame_count += 1

    cap.release()
    print(f"Procesados {frame_count} frames a {fps} FPS")
    return np.array(motion_signal), fps


# Detectar picos de respiración y generar datos para el análisis
def analyze_breathing(motion_signal, fps):
    # Filtrar la señal para eliminar ruido
    if len(motion_signal) < 10:  # Verificar que hay suficientes datos
        print("Error: Señal demasiado corta para analizar")
        return [], [], []

    filtered_signal = butter_bandpass_filter(motion_signal, FREQ_MIN, FREQ_MAX, fps)

    # Normalizar la señal para mejor visualización
    filtered_signal = (filtered_signal - np.min(filtered_signal)) / (np.max(filtered_signal) - np.min(filtered_signal))

    # Detección de picos (momentos de inhalación)
    min_dist = int(fps * 1.0)  # Al menos 1s entre picos
    threshold = 0.3  # Umbral para detección de picos

    try:
        peak_indices = peakutils.indexes(filtered_signal, thres=threshold, min_dist=min_dist)
        peak_times = peak_indices / fps
    except Exception as e:
        print(f"Error en la detección de picos: {e}")
        peak_indices = []
        peak_times = []

    return filtered_signal, peak_indices, peak_times


# Función para estimar respiraciones por minuto en un instante específico
def estimate_bpm_at_time(peak_times, current_time):
    # Filtrar picos hasta el tiempo actual
    peaks_so_far = [t for t in peak_times if t <= current_time]

    if len(peaks_so_far) < 2:
        if len(peaks_so_far) == 1 and current_time > peaks_so_far[0]:
            # Estimación con un solo pico (asumiendo periodo constante)
            time_since_peak = current_time - peaks_so_far[0]
            if time_since_peak > 0:
                return 60.0 / (time_since_peak * 2)  # Asumimos que el ciclo es el doble del tiempo transcurrido
        return 0.0  # No podemos estimar

    # Calcular intervalos entre picos consecutivos
    intervals = np.diff(peaks_so_far)

    # Usar los últimos 3 intervalos (o menos si no hay suficientes)
    recent_intervals = intervals[-min(3, len(intervals)):]

    # Promedio de intervalos recientes
    avg_interval = np.mean(recent_intervals)

    # Convertir a BPM
    if avg_interval > 0:
        return 60.0 / avg_interval
    return 0.0


# Función para verificar si hay una inhalación en un momento específico
def is_inhaling_at_time(peak_times, current_time, tolerance=0.2):
    for peak_time in peak_times:
        if abs(current_time - peak_time) <= tolerance:
            return True
    return False

# Función para generar la gráfica estática
def generate_static_graph(filtered_signal, peak_times, peak_indices, fps):
    # Graficar resultados estáticos
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(filtered_signal)) / fps, filtered_signal, label="Señal Filtrada")
    plt.scatter(peak_times, filtered_signal[peak_indices], color='red', label="Picos (Exhalaciones)")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Movimiento Normalizado")
    plt.title("Frecuencia Respiratoria Detectada")
    plt.legend()

    # Guardar gráfica estática
    static_graph_path = "respiracion_grafica.png"
    plt.savefig(static_graph_path, dpi=150)
    plt.close()
    print(f"Gráfica estática guardada: {static_graph_path}")


# Generar video de análisis de respiración con leyendas
def generate_breathing_analysis_video_with_legend(input_video_path, filtered_signal, peak_indices, peak_times, fps, output_path):
    # Extraer información del video original
    cap = cv2.VideoCapture(input_video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

    # Número total de frames a generar
    total_frames = int(len(filtered_signal) / fps * video_fps)

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / video_fps
        signal_idx = min(int(current_time * fps), len(filtered_signal) - 1)

        # Visualizar la señal de respiración en la parte inferior del video
        is_inhaling = is_inhaling_at_time(peak_times, current_time)
        bpm = estimate_bpm_at_time(peak_times, current_time)

        # Escribir leyendas sobre el video
        cv2.putText(frame, f"Exhalacion detectada: {'TRUE' if is_inhaling else 'FALSE'}", (10, height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_inhaling else (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"RpM: {bpm:.1f} RpM", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        print(f"Frame {frame_idx + 1}/{total_frames} - Exhalacion detectada: {is_inhaling} - RpM: {bpm:.1f} RpM")

        # Escribir el frame con las leyendas sobre el video
        out.write(frame)

    cap.release()
    out.release()
    print(f"Video de análisis generado: {output_path}")


# Función principal que integra todo el proceso
def process_video_and_generate_analysis(input_video_path, output_video_path=None):
    print(f"Procesando video: {input_video_path}")
    start_time = time.time()

    # Extraer señal de movimiento
    motion_signal, fps = extract_motion_signal(input_video_path)
    if motion_signal is None or len(motion_signal) == 0:
        print("Error: No se pudo extraer la señal de movimiento.")
        return

    print(f"Señal extraída: {len(motion_signal)} muestras a {fps} FPS")

    # Analizar respiración
    filtered_signal, peak_indices, peak_times = analyze_breathing(motion_signal, fps)

    # Generar video con análisis y leyendas
    if output_video_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = f"analisis_respiracion.mp4"

    generate_breathing_analysis_video_with_legend(input_video_path, filtered_signal, peak_indices, peak_times, fps, output_video_path)
    generate_static_graph(filtered_signal, peak_times, peak_indices, fps)

    elapsed_time = time.time() - start_time
    print(f"Proceso completado en {elapsed_time:.2f} segundos.")
    return output_video_path


# Ejemplo de uso
if __name__ == "__main__":
    # Ruta del video a analizar (cambiar según sea necesario)
    input_video = "C:\\Users\\crseg\\Desktop\\majo_video.mp4"

    # Procesar video y generar análisis
    output_video = process_video_and_generate_analysis(input_video)

    print(f"Video de análisis generado: {output_video}")
    print("¡Proceso completado!")
