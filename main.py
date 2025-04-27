import tkinter as tk
from tkinter import filedialog
from heartRate import getHeartRate
from respiratoryRate import getRespiratoryRate
from heartRateWebcam import getHeartRateWebcam

def obtenerBPMRPMVideo():
    filepath = filedialog.askopenfilename(title="Selecciona un video para calcular bpm y el rpm.")
    if filepath:
        print(f"Ruta completa: {filepath}")
        print(f"Nombre del archivo: {filepath.split('/')[-1]}")
        nombreArchivo = filepath.split('/')[-1]
        bpm_values, face_frames = getHeartRate(nombreArchivo, filepath)
        getRespiratoryRate(nombreArchivo, filepath, bpm_values, face_frames)


def obtenerBPMWebcam():
    getHeartRateWebcam()


# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Medidor de BPM y RPM")
ventana.geometry("300x100")  # Tamaño de la ventana

# Crear los botones
boton_rpm_video = tk.Button(ventana, text="Obtener BPM y RPM por video", command=obtenerBPMRPMVideo)
boton_bpm_webcam = tk.Button(ventana, text="Obtener BPM por webcam", command=obtenerBPMWebcam)


# Posicionar los botones en la ventana
boton_rpm_video.pack(pady=10)
boton_bpm_webcam.pack(pady=10)

# Ejecutar el loop principal
ventana.mainloop()
