import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.fft import fft, fftfreq, fftshift


# Frame rate of the camera in frames per second
frame_rate = 30.0  
sampling_interval = 1.0 / frame_rate  # Sampling interval in seconds

# Perform the Fourier transform
N = len(smooth_values)
y_fft = fft(smooth_values)
yf = fftfreq(N, sampling_interval)[:N//2]
y_fft_magnitude = 2.0/N * np.abs(y_fft[0:N//2])

# Generate time vector based on sampling interval
time_vector = np.arange(N) * sampling_interval


# Plot the original signal
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time_vector, smooth_values)
plt.title('Original Signal')
plt.xlabel('Time [s]')
plt.ylabel('x-position')
