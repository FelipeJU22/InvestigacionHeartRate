import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft


def butter_bandpass(lowcut, highcut, fs, order=5):
    if fs <= 0:
        raise ValueError("Sampling frequency must be greater than zero.")
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    if fs <= 0:
        return signal
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, signal, axis=0)


def moving_average(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')


def extract_rppg_signal(frames, fps, face_cascade):
    green_means = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            forehead = frame[y:y + h // 5, x:x + w]
            green_channel = forehead[:, :, 1]
            green_means.append(np.mean(green_channel))

    if len(green_means) > 0:
        filtered_signal = apply_bandpass_filter(green_means, 0.7, 2.5, fps)
        smoothed_signal = moving_average(filtered_signal, window_size=5)
        return smoothed_signal
    else:
        return None


def compute_heart_rate_per_second(rppg_signal, fps):
    if rppg_signal is None or len(rppg_signal) == 0 or fps <= 0:
        return None

    segment_length = fps
    heart_rates = []

    for i in range(0, len(rppg_signal) - segment_length, segment_length):
        segment = rppg_signal[i:i + segment_length]
        freqs = np.fft.fftfreq(len(segment), d=1 / fps)
        fft_values = np.abs(fft(segment))
        valid_indices = np.where((freqs > 0.7) & (freqs < 2.5))
        if len(valid_indices[0]) > 0:
            peak_freq = freqs[valid_indices][np.argmax(fft_values[valid_indices])]
            heart_rate = peak_freq * 60
            heart_rates.append(heart_rate)

    return heart_rates
