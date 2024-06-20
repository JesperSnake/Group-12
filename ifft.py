import math
import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt

nummers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,29,30]

#Normalizeren
def Normalize(data):

    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = ((data - min_val) / (max_val - min_val) - 0.5 ) * 2
    return normalized_data

#Fourier
def fourierfunctie(wormnummer):
    ############# TWEAK THIS
    Highest_freqs = 3
    wormnummer = wormnummer
    ############################

    bestandnaam = f'angles_{wormnummer}.txt'
    plaatje = f'angles_{wormnummer}.png'

    # Bestand openen

    og_list = []

    with open(bestandnaam, 'r') as input_bestand:
        content = input_bestand.read()
        trimmed_content = content.rstrip()
        content = trimmed_content
        content = content.split(' ')
        
        for value in content:
            og_list.append(float(value))


  

    og_list = Normalize(og_list)

    # Frame rate of the camera in frames per second
    frame_rate = 30.0  
    sampling_interval = 1.0 / frame_rate  # Sampling interval in seconds

    # Perform the Fourier transform
    N = len(og_list)
    y_fft = fft(og_list)
    yf = fftfreq(N, sampling_interval)[:N//2]
    y_fft_magnitude = 2.0/N * np.abs(y_fft[0:N//2])

    # Generate time vector based on sampling interval
    time_vector = np.arange(N) * sampling_interval


    #Bijknippen
    short_yfft = np.copy(y_fft)
    short_yfft = short_yfft[1::]
    short_mag = np.copy(y_fft_magnitude)
    short_mag = short_mag[1::]

    # FILTER
    magnitudes = np.abs(short_yfft)
    top_indices = np.argsort(magnitudes)[-Highest_freqs*2:]   


    filtered_y_fft = np.copy(y_fft)
    filtered_y_fft = np.zeros_like(y_fft)
    filtered_y_fft[top_indices] = y_fft[top_indices]


    # Perform the inverse Fourier transform
    reconstructed_signal = ifft(filtered_y_fft).real

    # Plot the original signal
    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    plt.plot(time_vector, og_list)
    plt.title('Original Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('x-position')

    # Plot the Fourier transform magnitude
    plt.subplot(4, 1, 2)
    plt.bar(yf[:N], y_fft_magnitude[:N], width=0.1)
    plt.title('Fourier Transform')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    

    plt.subplot(4, 1, 3)
    plt.bar(yf[:N//2], 2.0/N * np.abs(filtered_y_fft[:N//2]), width=0.1)
    plt.title('Filtered Fourier Transform')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')

    # Plot the reconstructed signal
    plt.subplot(4, 1, 4)
    plt.plot(time_vector, reconstructed_signal)
    plt.title('Reconstructed Signal from Inverse Fourier Transform')
    plt.xlabel('Time [s]')
    plt.ylabel('x-position')

    plt.tight_layout()
    plt.show()

fourierfunctie(2)
