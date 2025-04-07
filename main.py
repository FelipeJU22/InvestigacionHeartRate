import tkinter as tk
from tkinter import filedialog
from heartRate import getHeartRate
from respiratoryRate import getRespiratoryRate

def obtenerBPMVideo():
    filepath = filedialog.askopenfilename(title="Selecciona un video para BPM")
    if filepath:
        print(f"Ruta completa: {filepath}")
        print(f"Nombre del archivo: {filepath.split('/')[-1]}")
        nombreArchivo = filepath.split('/')[-1]
        getHeartRate(nombreArchivo, filepath)

def obtenerBPMWebcam():
    print("Aquí no hay archivo, usaría la webcam directamente.")  # Este no necesita abrir archivo

def obtenerRPMVideo():
    filepath = filedialog.askopenfilename(title="Selecciona un video para RPM")
    if filepath:
        print(f"Ruta completa: {filepath}")
        print(f"Nombre del archivo: {filepath.split('/')[-1]}")
        nombreArchivo = filepath.split('/')[-1]
        getRespiratoryRate(nombreArchivo, filepath)

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Medidor de BPM y RPM")
ventana.geometry("300x200")  # Tamaño de la ventana

# Crear los botones
boton_bpm_video = tk.Button(ventana, text="Obtener BPM por video", command=obtenerBPMVideo)
boton_bpm_webcam = tk.Button(ventana, text="Obtener BPM por webcam", command=obtenerBPMWebcam)
boton_rpm_video = tk.Button(ventana, text="Obtener RPM por video", command=obtenerRPMVideo)

# Posicionar los botones en la ventana
boton_bpm_video.pack(pady=10)
boton_bpm_webcam.pack(pady=10)
boton_rpm_video.pack(pady=10)

# Ejecutar el loop principal
ventana.mainloop()
