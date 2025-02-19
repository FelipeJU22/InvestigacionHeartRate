import cv2
import numpy as np
from PIL import Image

def motion_magnification(video_path, magnification_factor=10, low_pass_cutoff=0.1, high_pass_cutoff=0.1):
    # Leer el video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Inicializar la lista de frames procesados
    frames = []

    # Leer el video frame por frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    # Convertir los frames a escala de grises
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

    # Crear la lista de frames magnificados
    motion_frames = []

    for i in range(1, len(gray_frames)):
        # Substraer el movimiento de los frames
        frame_diff = cv2.absdiff(gray_frames[i], gray_frames[i - 1])

        # Aplicar un filtro de paso bajo y paso alto para el movimiento
        fft_frame = np.fft.fft2(frame_diff)
        fft_frame = np.fft.fftshift(fft_frame)

        # Filtrar las frecuencias
        rows, cols = fft_frame.shape
        center_row, center_col = rows // 2, cols // 2

        # Filtrar bajas y altas frecuencias
        fft_frame[:center_row - int(center_row * low_pass_cutoff), :] = 0
        fft_frame[center_row + int(center_row * low_pass_cutoff):, :] = 0
        fft_frame[:, :center_col - int(center_col * high_pass_cutoff)] = 0
        fft_frame[:, center_col + int(center_col * high_pass_cutoff):] = 0

        # Amplificar el movimiento
        fft_frame *= magnification_factor

        # Volver al espacio de imagen
        amplified_frame = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_frame)))

        # Convertir de vuelta a una imagen en escala de grises
        amplified_frame = np.uint8(np.clip(amplified_frame, 0, 255))

        # Agregar a la lista de frames procesados
        motion_frames.append(amplified_frame)

    return motion_frames

def create_gif_from_frames(frames, output_path, duration=100):
    # Convertir frames a PIL Images y luego guardarlos como un GIF
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(output_path, save_all=True, append_images=pil_frames[1:], optimize=False, duration=duration, loop=0)

# Usar la función
video_path = 'prueba.mp4'
motion_frames = motion_magnification(video_path, magnification_factor=10)

# Crear el GIF con los frames procesados
output_gif_path = 'output_motion_magnification.gif'
create_gif_from_frames(motion_frames, output_gif_path)
