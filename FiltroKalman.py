import numpy as np


class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_value=0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_value
        self.estimate_error = 1.0

    def update(self, measurement):
        # Predicción
        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_variance

        # Actualización
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error

        return self.estimate

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