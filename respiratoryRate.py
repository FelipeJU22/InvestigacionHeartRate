import cv2
import numpy as np
import peakutils
import matplotlib.pyplot as plt
from mediapipe.calculators import video
from scipy.signal import butter, filtfilt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
from datetime import datetime
import time

from FiltroKalman import KalmanFilter

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


def extract_motion_signal(video_path, roi=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or not cap.isOpened():
        print("Error: No se pudo leer el video o FPS inválido.")
        cap.release()
        return None, None

    # Determinar automáticamente una ROI si no se proporciona
    if roi is None:
        ret, first_frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el primer frame.")
            cap.release()
            return None, None
        height, width = first_frame.shape[:2]
        # Crear una ROI en el centro del torso (ajustar según sea necesario)
        roi = (width // 4, height // 3, width // 2, height // 3)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar el video

    motion_signal = []
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extraer región de interés
        x, y, w, h = roi
        roi_frame = frame[y:y + h, x:x + w]

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        # Aplicar un filtro Gaussiano para reducir ruido
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_frame is not None:
            # Calcular el flujo óptico o diferencia de frames
            diff = np.mean(np.abs(gray.astype(np.float32) - prev_frame))
            motion_signal.append(diff)

        prev_frame = gray.copy()

    cap.release()
    return np.array(motion_signal), fps


def analyze_breathing(motion_signal, fps):
    if len(motion_signal) < 10:
        print("Error: Señal demasiado corta para analizar")
        return [], [], []

    # Aplicar un filtro de mediana para eliminar outliers
    median_filtered = np.array(motion_signal)
    window_size = int(fps * 0.2)  # Ventana de 0.2 segundos
    if window_size % 2 == 0:
        window_size += 1  # Asegurar que sea impar
    if window_size > 1:
        try:
            from scipy.signal import medfilt
            median_filtered = medfilt(motion_signal, window_size)
        except:
            pass

    # Luego aplicar el filtro de paso banda
    filtered_signal = butter_bandpass_filter(median_filtered, FREQ_MIN, FREQ_MAX, fps)

    # Detrending para eliminar tendencias de largo plazo
    try:
        from scipy import signal
        filtered_signal = signal.detrend(filtered_signal)
    except:
        pass

    # Normalizar la señal
    filtered_signal = (filtered_signal - np.min(filtered_signal)) / (
                np.max(filtered_signal) - np.min(filtered_signal) + 1e-10)

    # Detección de picos mejorada
    min_dist = int(fps * 1.5)  # Ajustar según la frecuencia respiratoria mínima esperada
    threshold = 0.25  # Ajustar según la calidad de la señal

    try:
        # Usar peakutils con interpolación para mejor precisión
        peak_indices = peakutils.indexes(filtered_signal, thres=threshold, min_dist=min_dist, thres_abs=False)

        # Verificación adicional de picos con criterios fisiológicos
        peak_times = peak_indices / fps
        if len(peak_times) > 1:
            intervals = np.diff(peak_times)
            avg_interval = np.mean(intervals)
            if avg_interval < 1.0 or avg_interval > 10.0:  # Fuera del rango respiratorio normal
                print(f"Advertencia: Intervalo respiratorio promedio ({avg_interval:.2f}s) fuera de rango normal")
    except Exception as e:
        print(f"Error en la detección de picos: {e}")
        peak_indices = []
        peak_times = []

    return filtered_signal, peak_indices, peak_times


def detect_breathing_dual(filtered_signal, fps):
    # Detección de picos
    min_dist = int(fps * 1.5)
    threshold = 0.25
    peak_indices = peakutils.indexes(filtered_signal, thres=threshold, min_dist=min_dist)

    # Detección de cruces por cero (para fase respiratoria)
    zero_crossings = np.where(np.diff(np.signbit(filtered_signal - np.mean(filtered_signal))))[0]

    # Combinar detecciones
    phase_changes = []
    for i in range(0, len(zero_crossings) - 1, 2):
        if i + 1 < len(zero_crossings):
            # Identificar el comienzo de la inspiración (cruce positivo)
            phase_changes.append(zero_crossings[i])

    # Verificar consistencia entre métodos
    if len(peak_indices) > 0 and len(phase_changes) > 0:
        # Calcular la correlación entre las detecciones
        correlation = np.corrcoef(peak_indices, phase_changes[:len(peak_indices)])[0, 1] if len(peak_indices) == len(
            phase_changes) else 0
        print(f"Correlación entre métodos de detección: {correlation:.2f}")

    # Usar el método de detección más consistente
    final_indices = peak_indices if len(peak_indices) > len(phase_changes) else phase_changes
    final_times = final_indices / fps

    return filtered_signal, final_indices, final_times


def analyze_respiratory_frequency(filtered_signal, fps):
    # Calcular la transformada de Fourier
    from scipy.fft import fft, fftfreq

    # Asegurar que tengamos suficientes datos
    if len(filtered_signal) < fps * 10:  # Al menos 10 segundos
        print("Advertencia: Señal demasiado corta para análisis espectral confiable")
        return None

    n = len(filtered_signal)
    yf = fft(filtered_signal)
    xf = fftfreq(n, 1 / fps)

    # Solo analizar frecuencias positivas en el rango respiratorio
    positive_freqs = xf[:n // 2]
    power_spectrum = np.abs(yf[:n // 2])

    # Restringir al rango de frecuencias respiratorias (0.1 - 0.5 Hz)
    resp_range = (positive_freqs >= FREQ_MIN) & (positive_freqs <= FREQ_MAX)

    if np.any(resp_range):
        resp_freqs = positive_freqs[resp_range]
        resp_power = power_spectrum[resp_range]

        # Encontrar la frecuencia dominante
        dominant_freq_idx = np.argmax(resp_power)
        dominant_freq = resp_freqs[dominant_freq_idx]

        # Convertir a RPM
        rpm = dominant_freq * 60

        print(f"Frecuencia respiratoria dominante: {dominant_freq:.3f} Hz ({rpm:.1f} RPM)")
        return rpm
    else:
        print("No se encontraron frecuencias en el rango respiratorio")
        return None


def estimate_bpm_at_time(peak_times, current_time, kalman_filter):
    # Filtrar picos hasta el tiempo actual
    peaks_so_far = [t for t in peak_times if t <= current_time]

    if len(peaks_so_far) < 2:
        return kalman_filter.estimate  # Mantener el valor anterior si no hay suficientes picos

    # Calcular intervalos entre picos consecutivos
    intervals = np.diff(peaks_so_far)

    # Filtrar intervalos fuera de un rango razonable (por ejemplo, entre 2 y 10 segundos)
    valid_intervals = [interval for interval in intervals if 2.0 <= interval <= 10.0]

    if len(valid_intervals) == 0:
        return kalman_filter.estimate  # Mantener el valor anterior si no hay intervalos válidos

    # Calcular la RpM basada en el último intervalo válido
    current_interval = valid_intervals[-1]
    current_bpm = 60.0 / current_interval

    # Actualizar el filtro de Kalman con la nueva medición
    return kalman_filter.update(current_bpm)

# Función para verificar si hay una inhalación en un momento específico
def is_inhaling_at_time(peak_times, current_time, tolerance=0.15):
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
    static_graph_path = "ResultadoRespiratory/respiracion_grafica.png"
    plt.savefig(static_graph_path, dpi=150)
    plt.close()
    print(f"Gráfica estática guardada: {static_graph_path}")

# Generar video de análisis de respiración con leyendas
def generate_breathing_analysis_video_with_legend(input_video_path, filtered_signal, peak_indices, peak_times, fps, output_path, kalman_filter):
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
        bpm = estimate_bpm_at_time(peak_times, current_time, kalman_filter)

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


def process_video_and_generate_analysis(input_video_path, bpm_values, face_frames, output_video_path=None):
    print(f"Procesando video: {input_video_path}")
    start_time = time.time()

    # Extraer señal de movimiento con ROI automática
    motion_signal, fps = extract_motion_signal(input_video_path)
    if motion_signal is None or len(motion_signal) == 0:
        print("Error: No se pudo extraer la señal de movimiento.")
        return

    print(f"Señal extraída: {len(motion_signal)} muestras a {fps} FPS")

    # Analizar respiración con método mejorado
    filtered_signal, peak_indices, peak_times = analyze_breathing(motion_signal, fps)

    # Validar la detección con análisis de frecuencia
    rpm_from_fft = analyze_respiratory_frequency(filtered_signal, fps)

    # Inicializar el filtro de Kalman adaptativo
    kalman_filter = AdaptiveKalmanFilter(process_variance=0.1, measurement_variance=1.0,
                                         initial_value=rpm_from_fft if rpm_from_fft else 15.0)

    # Calcular métricas de calidad
    if len(peak_times) > 1:
        intervals = np.diff(peak_times)
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        cv_interval = std_interval / avg_interval if avg_interval > 0 else float('inf')

        print(f"Métricas de calidad:")
        print(f"- Intervalo respiratorio promedio: {avg_interval:.2f}s")
        print(f"- Desviación estándar: {std_interval:.2f}s")
        print(f"- Coeficiente de variación: {cv_interval:.2f}")
        print(f"- Frecuencia respiratoria estimada: {60 / avg_interval:.1f} RPM")
        print(f"- Frecuencia desde FFT: {rpm_from_fft:.1f} RPM" if rpm_from_fft else "- FFT no disponible")

    # Generar video con análisis y leyendas mejoradas
    if output_video_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = f'ResultadoRespiratory/Resultado_{timestamp}.mp4'

    generate_enhanced_breathing_analysis_video(input_video_path, filtered_signal, peak_indices, peak_times, fps,
                                               output_video_path, kalman_filter, bpm_values, face_frames)
    generate_static_graph(filtered_signal, peak_times, peak_indices, fps)

    elapsed_time = time.time() - start_time
    print(f"Proceso completado en {elapsed_time:.2f} segundos.")
    return output_video_path


def generate_enhanced_breathing_analysis_video(input_video_path, filtered_signal, peak_indices, peak_times, fps,
                                               output_path, kalman_filter, bpm_values, face_frames):
    # Extraer información del video original
    cap = cv2.VideoCapture(input_video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Crear espacio adicional para la gráfica
    output_height = height + 200  # Añadir 200 píxeles para la gráfica
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, output_height))

    # Calcular el total de frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Establecer el historial de RPM para la gráfica
    rpm_history = []
    time_history = []

    # Precalcular los valores de la señal filtrada a lo largo del tiempo
    signal_times = np.arange(len(filtered_signal)) / fps

    # Historia de la respiración para visualización
    breathing_history = []
    history_length = int(width * 0.8)  # 80% del ancho para el historial

    # Procesar cada frame
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Calcular el tiempo actual
        current_time = frame_idx / video_fps

        # Encontrar el índice más cercano en nuestra señal filtrada
        closest_idx = min(int(current_time * fps), len(filtered_signal) - 1)

        # Determinar si estamos en un pico de respiración
        is_inhaling = is_inhaling_at_time(peak_times, current_time)

        # Estimar RPM con el filtro de Kalman
        bpm = estimate_bpm_at_time(peak_times, current_time, kalman_filter)
        rpm_history.append(bpm)
        time_history.append(current_time)

        # Limitar el historial para mostrar solo los últimos 30 segundos
        while time_history and time_history[0] < current_time - 30:
            time_history.pop(0)
            rpm_history.pop(0)

        # Añadir el valor actual a la historia de respiración
        breathing_value = filtered_signal[closest_idx] if closest_idx < len(filtered_signal) else 0
        breathing_history.append(breathing_value)
        if len(breathing_history) > history_length:
            breathing_history.pop(0)

        # Crear una imagen combinada (video original + gráficas)
        combined_image = np.zeros((output_height, width, 3), dtype=np.uint8)
        combined_image[:height, :] = frame

        # Área gráfica (fondo negro)
        combined_image[height:, :] = [0, 0, 0]

        # Dibujar la señal de respiración
        graph_height = 100
        graph_y_offset = height + 50

        # Dibujar línea base
        cv2.line(combined_image,
                 (50, graph_y_offset),
                 (width - 50, graph_y_offset),
                 (100, 100, 100), 1)

        # # Dibujar historia de respiración
        for i in range(1, len(breathing_history)):
            x1 = width - 50 - history_length + i - 1
            y1 = graph_y_offset - int(breathing_history[i - 1] * graph_height)
            x2 = width - 50 - history_length + i
            y2 = graph_y_offset - int(breathing_history[i] * graph_height)

            # Color verde para inhalación, rojo para exhalación
            color = (0, 255, 0) if is_inhaling else (255, 0, 0)
            if x1 > 0 and x2 > 0 and x1 < width and x2 < width:
                cv2.line(combined_image, (x1, y1), (x2, y2), color, 2)


        # Dibujar RPM actual en forma de texto y un indicador visual
        cv2.putText(combined_image, f"RpM: {bpm:.1f}", (width - 175, height + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Indicador visual de calidad de detección
        quality_color = (0, 255, 0)  # Verde por defecto (buena)


        # Visualizar fase de respiración (inhalación/exhalación)
        phase_text = "Inhalacion" if is_inhaling else "Exhalacion"
        cv2.putText(combined_image, phase_text, (width // 2 - 80, height + 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if is_inhaling else (0, 0, 255), 2)

        # Añadir una superposición visual de la onda respiratoria
        # Dibujar línea de tiempo actual en la gráfica
        timeline_x = width - 50 - history_length + len(breathing_history) - 1
        if timeline_x > 0 and timeline_x < width:
            cv2.line(combined_image, (timeline_x, height + 10), (timeline_x, height + 190), (255, 255, 255), 1)

        # Dibujar marcadores de picos previos y futuros
        for peak_time in peak_times:
            peak_x = int(width - 50 - history_length + (
                    peak_time - (current_time - len(breathing_history) / video_fps)) * video_fps)
            if 0 < peak_x < width:
                cv2.circle(combined_image, (peak_x, graph_y_offset - graph_height // 2), 5, (0, 0, 255), -1)
        if bpm_values and frame_idx < len(bpm_values):
            bpm_current = bpm_values[frame_idx]
            cv2.putText(combined_image, f"BPM: {bpm_current:.1f}", (width - 175, height + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # Mostrar el rostro magnificado si está disponible
        if face_frames and frame_idx < len(face_frames):
            face_frame_resized = cv2.resize(face_frames[frame_idx], (120, 120))  # Ajustar tamaño
            combined_image[height + 60:height + 180, 10:130] = face_frame_resized  # Esquina inferior izquierda

        # Escribir el frame combinado
        out.write(combined_image)

        # Mostrar progreso
        if frame_idx % 30 == 0 or frame_idx == total_frames - 1:
            print(f"Procesando frame {frame_idx + 1}/{total_frames} - RpM: {bpm:.1f}")


    cap.release()
    out.release()
    print(f"Video de análisis mejorado generado: {output_path}")


class AdaptiveKalmanFilter:
    def __init__(self, initial_value=15.0, process_variance=0.1, measurement_variance=1.0):
        self.estimate = initial_value
        self.error_covariance = 1.0
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.measurement_history = []
        self.min_possible_value = 1.0  # 6 RPM mínimo fisiológico
        self.max_possible_value = 30.0  # 30 RPM máximo fisiológico normal

    def update(self, measurement):
        # Restringir mediciones a rangos fisiológicos
        measurement = max(self.min_possible_value, min(self.max_possible_value, measurement))

        # Añadir medición al historial
        self.measurement_history.append(measurement)
        if len(self.measurement_history) > 10:
            self.measurement_history.pop(0)

        # Adaptar varianzas basado en consistencia reciente
        if len(self.measurement_history) > 3:
            recent_variance = np.var(self.measurement_history)
            if recent_variance < 1.0:  # Mediciones muy consistentes
                self.process_variance = 0.05
            elif recent_variance > 5.0:  # Mediciones muy variables
                self.process_variance = 0.2

        # Predicción
        prediction = self.estimate
        prediction_error_covariance = self.error_covariance + self.process_variance

        # Actualización
        kalman_gain = prediction_error_covariance / (prediction_error_covariance + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)

        # Restringir la estimación a rangos fisiológicos
        self.estimate = max(self.min_possible_value, min(self.max_possible_value, self.estimate))

        # Actualizar error de covarianza
        self.error_covariance = (1 - kalman_gain) * prediction_error_covariance

        return self.estimate

# Ejemplo de uso
def getRespiratoryRate(videoName, videoPath, bpm_values=None, face_frames = None):
    # Ruta del video a analizar (cambiar según sea necesario)
    test = video
    input_video = videoPath
    # Procesar video y generar análisis
    output_video = process_video_and_generate_analysis(input_video, bpm_values, face_frames)

    print(f"Video de análisis generado: {output_video}")
    print("¡Proceso completado!")






