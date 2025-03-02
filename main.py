import cv2
import numpy as np
from heartRate import extract_rppg_signal, compute_heart_rate_per_second, apply_bandpass_filter


def motion_magnification(videoPath, outputPath, magnificationFactor=20, lowFreq=0.4, highFreq=3.0):
    cap = cv2.VideoCapture(videoPath)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        print("Error: FPS del video no válido.")
        cap.release()
        return None

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outputPath, fourcc, fps, (width, height))

    frames = []
    originalFrames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yuvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        frames.append(yuvFrame[:, :, 0])
        originalFrames.append(frame)
    cap.release()

    print(f"Procesando {len(frames)} frames de video...")

    frames = np.array(frames, dtype=np.float32)
    filteredFrames = apply_bandpass_filter(frames, lowFreq, highFreq, fps)
    amplifiedFrames = frames + magnificationFactor * filteredFrames
    amplifiedFrames = np.clip(amplifiedFrames, 0, 255).astype(np.uint8)

    print("Extrayendo señal rPPG...")
    rppg_signal = extract_rppg_signal(originalFrames, fps, face_cascade)
    print("Calculando frecuencia cardiaca...")
    heart_rates = compute_heart_rate_per_second(rppg_signal, fps)

    # Imprimir todos los valores de BPM
    print("\n--- VALORES DE BPM POR SEGUNDO ---")
    for second, bpm in enumerate(heart_rates):
        print(f"Segundo {second + 1}: {bpm:.2f} BPM")
    print("--------------------------------\n")

    # Calcula el BPM promedio
    if heart_rates:
        avg_bpm = sum(heart_rates) / len(heart_rates)
        print(f"BPM promedio: {avg_bpm:.2f}")

    for i in range(len(amplifiedFrames)):
        yuvFrame = cv2.cvtColor(originalFrames[i], cv2.COLOR_BGR2YUV)
        yuvFrame[:, :, 0] = amplifiedFrames[i]
        outputFrame = cv2.cvtColor(yuvFrame, cv2.COLOR_YUV2BGR)

        if heart_rates and i // fps < len(heart_rates):
            current_bpm = heart_rates[i // fps]
            text = f"HR: {current_bpm:.2f} BPM"
            # Solo imprimir cuando cambia el segundo
            if i % fps == 0:
                print(f"Frame {i} - {text}")

            (textWidth, textHeight), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(outputFrame, (45, 20), (45 + textWidth + 10, 35 + textHeight + 20), (0, 0, 0), -1)
            cv2.putText(outputFrame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(outputFrame)
        cv2.imshow("Output Video", outputFrame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()

    print("\nProcesamiento completado!")
    print(f"Video guardado en: {outputPath}")
    return outputPath


if __name__ == "__main__":
    videoPath = "C:\\Users\\crseg\\Desktop\\majo_video.mp4"
    outputVideoPath = 'output_motion_magnification.avi'
    motion_magnification(videoPath, outputVideoPath)