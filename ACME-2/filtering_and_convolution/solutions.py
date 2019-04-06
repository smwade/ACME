# Name this file solutions.py.
"""Volume II Lab 10: Fourier II (Filtering and Convolution)
Sean Wade
Math 321
Nov 17, 2015
"""

import numpy as np
import scipy as sp
import scipy.fftpack as fftpack
from scipy.io import wavfile
import IPython.display as play
from matplotlib import pyplot as plt
import cmath
import wave

class Signal(object):
    def __init__(self, sample_rate, sample_array):
        """Initialize the a object of type Signal with its sample_rate and sample_array
        """ 
        self.rate = sample_rate
        self.signal = sample_array
        self.dtf = self.calculate_DFT()
    
    def __add__(self, other):
        if self.rate != other.rate:
            raise ValueError("The sample rates are not the same.")
        new_signal = self.signal + other.signal
        return Signal(self.rate, new_signal)
    
    def plot(self, DFT=False):
        if DFT is True:
            self.calculate_DFT()
            x_vals = np.arange(1, len(self.dft)+1, 1) * 1.
            x_vals = x_vals / len(self.signal)
            x_vals = x_vals * self.rate
            plt.plot(x_vals, np.abs(self.dft))
        else:
            plt.plot(self.signal)
         
    def write_to_file(self, filename):
        wavfile.write(filename, self.rate, self.signal)
        
    def calculate_DFT(self):
        self.dft = fftpack.fft(self.signal)


# Problem 1: Implement this function.
def clean_signal(outfile='prob1.wav'):
    """Clean the 'Noisysignal2.wav' file. Plot the resulting sound
    wave and write the resulting sound to the specified outfile."""

    # Load the noisy .wav file
    rate, signal = wavfile.read('./Fourier2/Noisysignal2.wav')
    bad_signal = Signal(rate, signal)

    # Plot the original sound and its FFT
    plt.figure(1).suptitle("Original Sound")
    plt.subplot(121)
    bad_signal.plot()
    plt.subplot(122)
    bad_signal.plot(True)
    plt.show()

    # Cut out all the bad frequencies from the FFT
    fsig = sp.fft(bad_signal.signal, axis=0)
    for j in xrange(14999, 50000):
        fsig[j] = 0
        fsig[-j] = 0

    # inverse FFT back and scale
    newsig = sp.ifft(fsig)
    newsig = sp.real(newsig)
    newsig = sp.int16(newsig / sp.absolute(newsig).max() * 32767)
    clean_signal = Signal(rate, newsig)

    # Plot the clean sound and its FFT
    plt.figure(2).suptitle("Clean Sound")
    plt.subplot(121)
    clean_signal.plot(False)
    plt.subplot(122)
    clean_signal.plot(True)
    plt.show()

    bad_signal.plot()
    clean_signal.plot()
    plt.show()

    clean_signal.write_to_file("clean_2.wav")
    # FDR Fear Speach
    

# Problem 3: Implement this function.
def convolve(source='chopin.wav', pulse='balloon.wav', outfile='prob3.wav'):
    """Convolve the specified source file with the specified pulse file, then
    write the resulting sound wave to the specified outfile.
    """

    input_rate, input_signal = wavfile.read(source)
    ballon_rate, ballon_signal = wavfile.read(pulse)
    input_signal_zeros = np.hstack((input_signal[:,1], np.zeros(5*44100)))
    full_signal_len = len(input_signal_zeros)
    num_of_zeros_for_ballon = full_signal_len - len(ballon_signal)
    ballon_middle = len(ballon_signal) / 2
    balloon_signal_zeros = np.hstack([ballon_signal[:ballon_middle,1], np.zeros((num_of_zeros_for_ballon)), ballon_signal[ballon_middle:, 1]])
    full_balloon = balloon_signal_zeros
    full_chopin = input_signal_zeros
    fourier_convolution = np.multiply(sp.fft(full_chopin), sp.fft(full_balloon))
    result = sp.ifft(fourier_convolution)
    result = sp.real(result)
    result = sp.int16(result / sp.absolute(result).max() * 32767)
    final = Signal(input_rate, result.real)
    final.write_to_file(outfile)


# Problem 4: Implement this function.
def white_noise(outfile='prob4.wav'):
    """Generate some white noise, write it to the specified outfile,
    and plot the spectrum (DFT) of the signal.
    """
    samplerate = 44100
    noise = sp.int16(sp.random.randint(-32767, 32767, samplerate*10))
    N = Signal(samplerate, noise)
    N.write_to_file(outfile)
    N.plot(True)
