from datetime import datetime
import numpy as np
import cv2
import sys
from cvzone.FaceDetectionModule import FaceDetector
import cvzone
import time
from collections import deque

class BPMFilter:
    """
    Clase avanzada para filtrar valores de BPM con múltiples etapas:
    1. Detección de outliers estadísticos (Z-score adaptativo)
    2. Filtro de mediana para robustez
    3. Filtro EMA adaptativo para suavizado
    4. Filtro Savitzky-Golay opcional para tendencias suaves
    """
    def __init__(self, median_window=7, ema_alpha=0.25, use_savgol=True):
        """
        Args:
            median_window: Tamaño de ventana para filtro de mediana (7-11 recomendado)
            ema_alpha: Factor inicial de suavizado EMA (0.15-0.35 recomendado)
            use_savgol: Activar filtro Savitzky-Golay adicional
        """
        self.median_window = median_window if median_window % 2 == 1 else median_window + 1
        self.ema_alpha = ema_alpha
        self.use_savgol = use_savgol
        
        # Buffers para diferentes filtros
        self.median_buffer = deque(maxlen=self.median_window)
        self.raw_history = deque(maxlen=20)  # Para detección de outliers
        self.filtered_history = deque(maxlen=10)  # Para Savitzky-Golay
        
        # Estado del filtro EMA
        self.ema_value = None
        self.ema_variance = None  # Para adaptabilidad
        
        # Límites fisiológicos
        self.min_bpm = 40.0
        self.max_bpm = 180.0
        
        # Parámetros adaptativos
        self.z_threshold = 2.5  # Umbral para detección de outliers
        self.alpha_min = 0.1    # Alpha mínimo (más suavizado)
        self.alpha_max = 0.5    # Alpha máximo (más reactivo)
        
    def _is_outlier(self, value):
        """Detecta outliers usando Z-score adaptativo"""
        if len(self.raw_history) < 5:
            return False
        
        history_array = np.array(list(self.raw_history))
        mean = np.mean(history_array)
        std = np.std(history_array)
        
        if std < 1.0:  # Evitar división por cero
            return False
        
        z_score = abs((value - mean) / std)
        return z_score > self.z_threshold
    
    def _adaptive_alpha(self):
        """Ajusta alpha del EMA según la variabilidad reciente"""
        if len(self.filtered_history) < 3:
            return self.ema_alpha
        
        # Calcular variabilidad reciente
        recent_values = np.array(list(self.filtered_history))
        variance = np.var(recent_values)
        
        # Si hay mucha variabilidad, aumentar alpha (más reactivo)
        # Si hay poca variabilidad, disminuir alpha (más suavizado)
        if variance > 50:  # Alta variabilidad
            return min(self.alpha_max, self.ema_alpha * 1.3)
        elif variance < 10:  # Baja variabilidad
            return max(self.alpha_min, self.ema_alpha * 0.8)
        else:
            return self.ema_alpha
    
    def _savitzky_golay_filter(self):
        """Aplica filtro Savitzky-Golay para suavizado adicional"""
        if len(self.filtered_history) < 5:
            return list(self.filtered_history)[-1] if self.filtered_history else None
        
        # Usar los últimos valores para el filtro
        window = min(7, len(self.filtered_history))
        if window % 2 == 0:
            window -= 1
        
        try:
            from scipy.signal import savgol_filter
            values = np.array(list(self.filtered_history))
            smoothed = savgol_filter(values, window, 2)  # Polinomio grado 2
            return smoothed[-1]
        except:
            # Si scipy no está disponible, devolver el último valor
            return list(self.filtered_history)[-1]
    
    def add_measurement(self, bpm_raw):
        """
        Procesa una nueva medición a través de múltiples filtros
        """
        # Validación básica
        if not np.isfinite(bpm_raw):
            return self.ema_value if self.ema_value is not None else np.nan
        
        # Limitar al rango fisiológico
        bpm_clamped = np.clip(bpm_raw, self.min_bpm, self.max_bpm)
        
        # Agregar al historial de valores crudos
        self.raw_history.append(bpm_clamped)
        
        # ETAPA 1: Detección de outliers
        if self._is_outlier(bpm_clamped):
            # Si es outlier, usar el último valor filtrado válido
            if self.ema_value is not None:
                return self.ema_value
            else:
                bpm_clamped = np.median(list(self.raw_history)) if len(self.raw_history) > 0 else bpm_clamped
        
        # ETAPA 2: Filtro de mediana (robusto a outliers)
        self.median_buffer.append(bpm_clamped)
        if len(self.median_buffer) >= 3:
            bpm_median = np.median(list(self.median_buffer))
        else:
            bpm_median = bpm_clamped
        
        # ETAPA 3: Filtro EMA adaptativo
        current_alpha = self._adaptive_alpha()
        
        if self.ema_value is None:
            self.ema_value = bpm_median
        else:
            self.ema_value = current_alpha * bpm_median + (1 - current_alpha) * self.ema_value
        
        # Agregar al historial filtrado
        self.filtered_history.append(self.ema_value)
        
        # ETAPA 4: Savitzky-Golay (opcional, para suavizado final)
        if self.use_savgol and len(self.filtered_history) >= 5:
            final_value = self._savitzky_golay_filter()
        else:
            final_value = self.ema_value
        
        return final_value
    
    def get_current_value(self):
        """Retorna el valor actual filtrado"""
        return self.ema_value if self.ema_value is not None else np.nan
    
    def reset(self):
        """Reinicia todos los filtros"""
        self.median_buffer.clear()
        self.raw_history.clear()
        self.filtered_history.clear()
        self.ema_value = None
        self.ema_variance = None


def getHeartRate(videoName, videoPath):
    # Parámetros base del ROI (independientes de orientación del video)
    roiWidth = 160
    roiHeight = 120
    videoChannels = 3
    videoFrameRate = 15  # usado para banda

    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {videoPath}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputVideoPath = f"ResultadoHeartRate/Resultado_{timestamp}.avi"

    detector = FaceDetector()

    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Número total de frames: {totalFrames}")

    fpsIn = cap.get(cv2.CAP_PROP_FPS)
    if not fpsIn or fpsIn <= 0:
        fpsIn = 30.0  # fallback razonable

    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Detectar orientación (horizontal/vertical) y usar tamaño real para la salida
    isPortrait = frameHeight > frameWidth

    # VideoWriter con el mismo tamaño que el video de entrada (funciona para vertical u horizontal)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(outputVideoPath, fourcc, fpsIn, (frameWidth, frameHeight))

    # Parámetros de magnificación de color
    levels = 3
    alpha = 170
    minFrequency = 1.0
    maxFrequency = 2.0
    bufferSize = 150
    bufferIndex = 0

    # Helpers
    def buildGauss(frame, levelsCount):
        pyramid = [frame]
        for _ in range(levelsCount):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid

    def reconstructFrame(pyramid, index, levelsCount):
        filteredFrame = pyramid[index]
        for _ in range(levelsCount):
            filteredFrame = cv2.pyrUp(filteredFrame)
        filteredFrame = filteredFrame[:roiHeight, :roiWidth]
        return filteredFrame

    # UI dinámico (proporcional al tamaño del frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = max(frameWidth, frameHeight) / 800.0
    fontColor = (0, 0, 0)
    boxColor = (0, 255, 0)

    # Ubicaciones relativas (x,y) como fracciones del tamaño
    loadingTextLocation = (int(0.04 * frameWidth), int(0.08 * frameHeight))
    bpmTextLocation = (int(0.5 * frameWidth), int(0.08 * frameHeight))
    fpsTextLocation = (int(0.04 * frameWidth), int(0.92 * frameHeight))

    # Inicialización pirámide Gaussiana
    firstFrame = np.zeros((roiHeight, roiWidth, videoChannels), dtype=np.float32)
    firstGauss = buildGauss(firstFrame, levels + 1)[levels]
    videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels), dtype=np.float32)
    fourierTransformAvg = np.zeros((bufferSize), dtype=np.float32)

    # Filtro pasa banda
    frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
    mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

    # Variables de BPM
    bpmCalculationFrequency = 10
    bpmBufferIndex = 0
    bpmBufferSize = 10
    bpmBuffer = np.zeros((bpmBufferSize), dtype=np.float32)
    bpmValues = []  # se mantendrá alineado a frames (np.nan si no hay valor usable aún)

    # *** NUEVO: Instanciar el filtro de BPM mejorado ***
    # Configuración recomendada para máximo suavizado:
    # - median_window=7 o 9: Más resistente a outliers
    # - ema_alpha=0.2: Más suavizado (0.15-0.25 para videos con ruido)
    # - use_savgol=True: Activa suavizado adicional
    bpm_filter = BPMFilter(median_window=9, ema_alpha=0.2, use_savgol=True)

    i = 0
    ptime = 0.0
    faceFrames = []

    # Control de visualización: SOLO mostrar BPM después de 3 s
    frameIndex = 0
    delaySeconds = 3.0

    # Helper para formatear BPM aunque no sea finito
    def formatBpm(value):
        return f"{value:.1f}" if value is not None and np.isfinite(value) else "--"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frameDraw = frame.copy()
        ftime = time.time()
        fps = 1.0 / max(1e-6, (ftime - ptime))
        ptime = ftime

        elapsedSeconds = frameIndex / float(fpsIn)

        # FPS on-screen
        cv2.putText(frameDraw, f"FPS: {int(fps)}", fpsTextLocation, font, fontScale, fontColor, 2, cv2.LINE_AA)

        frameDetected, bboxs = detector.findFaces(frame, draw=False)

        bpmValue = np.nan  # valor por defecto por frame

        if bboxs:
            x, y, w, h = bboxs[0]["bbox"]

            # Asegurarse de que el bbox esté dentro del frame
            x = max(0, x)
            y = max(0, y)
            w = max(1, w)
            h = max(1, h)
            if x + w > frameWidth:
                w = frameWidth - x
            if y + h > frameHeight:
                h = frameHeight - y

            cv2.rectangle(frameDraw, (x, y, w, h), (255, 0, 255), 2)

            detectionFrame = frame[y : y + h, x : x + w]
            if detectionFrame.size != 0:
                detectionFrame = cv2.GaussianBlur(detectionFrame, (5, 5), 0)
                detectionFrame = cv2.resize(detectionFrame, (roiWidth, roiHeight))
                detectionFloat = detectionFrame.astype(np.float32)

                # Pirámide Gaussiana
                videoGauss[bufferIndex] = buildGauss(detectionFloat, levels + 1)[levels]
                fourierTransform = np.fft.fft(videoGauss, axis=0)

                # Pasa banda
                fourierTransform[~mask] = 0

                # Pulso (se calcula normalmente, pero NO condiciona la visualización)
                if bufferIndex % bpmCalculationFrequency == 0:
                    i += 1
                    # Promedio del canal G
                    for buf in range(bufferSize):
                        fourierTransformAvg[buf] = np.real(fourierTransform[buf][:, :, 1]).mean()
                    hz = frequencies[int(np.argmax(fourierTransformAvg))]
                    bpm_raw = 60.0 * hz
                    
                    # *** NUEVO: Aplicar filtros al BPM crudo ***
                    bpm_filtered = bpm_filter.add_measurement(bpm_raw)
                    
                    bpmBuffer[bpmBufferIndex] = bpm_filtered
                    bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

                # Amplificar
                filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
                filtered *= alpha

                # Reconstrucción
                filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
                outputFrame = detectionFloat + filteredFrame
                outputFrame = cv2.convertScaleAbs(outputFrame)
                faceFrames.append(outputFrame.copy())

                bufferIndex = (bufferIndex + 1) % bufferSize

                # Mini-panel ROI
                panel = cv2.resize(outputFrame, (roiWidth // 2, roiHeight // 2))
                panelH, panelW = panel.shape[:2]
                top = 0
                left = max(0, frameWidth - panelW)
                frameDraw[top : top + panelH, left : left + panelW] = panel

                # *** MODIFICADO: Usar el valor filtrado actual ***
                bpmValue = bpm_filter.get_current_value()

        # --- LÓGICA DE VISUALIZACIÓN ---
        if elapsedSeconds < delaySeconds:
            # Siempre mostrar "Calculando..." durante los primeros 3 segundos
            cvzone.putTextRect(
                frameDraw,
                "Calculando...",
                loadingTextLocation,
                font=font,
                scale=fontScale,
                colorR=fontColor,
                colorB=boxColor,
            )
        else:
            # A partir de los 3s, SIEMPRE mostrar el rótulo BPM, aunque el valor sea NaN
            cvzone.putTextRect(
                frameDraw,
                f"BPM: {formatBpm(bpmValue)}",
                bpmTextLocation,
                font=font,
                scale=fontScale,
                colorR=fontColor,
                colorB=boxColor,
            )

        # Guardar en el video y registrar el valor (alineado a frames)
        out.write(frameDraw)
        bpmValues.append(bpmValue if np.isfinite(bpmValue) else np.nan)

        # avanzar índice de frame
        frameIndex += 1

    # Liberar recursos antes de devolver
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return bpmValues, faceFrames