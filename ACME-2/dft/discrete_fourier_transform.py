# coding: utf-8

# #Volume II Lab 9
# ##Sean Wade
# 

# <b>Lab Objective</b>: The analysis of periodic functions has many applications in
# pure and applied mathematics, especially in settings dealing with sound waves. The
# Fourier transform provides a way to analyze such periodic functions. In this lab,
# we implement the discrete Fourier transform and explore digital audio signals.

# In[110]:

import numpy as np
import scipy as sp
import scipy.fftpack as fftpack
from scipy.io import wavfile
import IPython.display as play
from matplotlib import pyplot as plt
import cmath
import wave


# In[111]:


# ###Digital Audio Signals
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  There are two components of a digital audio signal: samples from the soundwave
# and a sample rate. These correspond to the amplitude and frequency, respectively.
# A sample is a measurement of the amplitude of the wave at an instant in time.
# To see why the sample rate is necessary, consider an array with samples from a
# soundwave. If we do not know how frequently those samples were collected then we
# can arbitrarily stretch or compress the soundwave to make a variety of sounds.
# However, if we know at what rate the samples were taken, then we can construct
# the wave exactly as it was recorded. In most applications, this sample rate will be
# measured in number of samples taken per second. The standard rate for high quality
# audio is 44100 equally spaced samples per second.

# In[125]:

class Signal(object):
    def __init__(self, sample_rate, sample_array):
        """Initialize the a object of type Signal with its sample_rate and sample_array
        """ 
        self.rate = sample_rate
        self.signal = np.int16(sample_array)
        self.dtf = None
    
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
            plt.plot(x_vals, self.dft)
        else:
            plt.plot(self.signal)
        plt.show()
        
    def write_to_file(self, filename):
        wavfile.write(filename, self.rate, self.signal)
        
    def calculate_DFT(self):
        self.dft = fftpack.fft(self.signal)


# In[115]:

def generate_note(frequency=440, duration=5):
    """Return an instance of the Signal class corresponding to the desired
    soundwave. Sample at a rate of 44100 samples per second.
    """
    samplerate = 44100
    wave_func = lambda x: np.sin(2*np.pi*x*frequency)
    stepsize = 1./samplerate
    sample_points = np.arange(0, duration, stepsize)
    samples = wave_func(sample_points)
    scaled_samples = np.int16(samples*32767)
    return Signal(samplerate, scaled_samples)


# Using generate_note we can create a sound just by a sample of the frequency and its sample rate.
# In[117]:

def my_fft(x):
    """Takes a audio ample and returns an array of calculated coefficients.
    x (np.array) : the sampled audio
    """
    n = len(x)
    DFT_matrix = np.ones((n,n), dtype=complex)
    for i in xrange(1,n):
        for j in xrange(1,n):
            DFT_matrix[i][j] = cmath.exp((-2*np.pi*1j*i*j)/n)
    return np.dot(DFT_matrix, x)


# To create a chord we take multiple signals and add them together.  We must be carefull that there is the same sample rate and the samples are properly scaled.

# In[118]:

def make_chord(note_list, durration, filename):
    """Takes a list of notes and makes a chord of length durration.
    note_list (list): List of notes for chord (all caps).
    durration (float): The lenght of the desired signal.
    """
    note_dict = {'A':440, 'B': 493.88, 'C': 523.25, 'D': 587.33, 'E': 659.25, 'F': 698.46, 'G': 783.99}
    sample_rate = 44100
    wave_function = lambda x, frequency: np.sin(2 * np.pi * x * frequency)
    step_size = 1. / sample_rate
    sample_points = np.arange(0,durration,step_size)
    samples = 0
    for note in note_list:
        samples += wave_function(sample_points, note_dict[note])
    scaled_samples = np.int16(samples * 10000 / len(note_list))
    chord = Signal(sample_rate, scaled_samples)
    chord.write_to_file(filename)
    return chord


# ### Some Example Chords and Their DFT

# #### C Major

# In[119]:

make_chord(['C', 'E', 'G'], 3, 'C_Major.wav')
rate, signal = wavfile.read('C_Major.wav')
C = Signal(rate, signal)
C.plot(True)


# #### F Major

# In[120]:

make_chord(['F', 'A', 'C'], 3, 'F_Major.wav')
rate, signal = wavfile.read('F_Major.wav')
F = Signal(rate, signal)
F.plot(True)


# #### G Major

# In[121]:

make_chord(['G', 'B', 'D'], 3, 'G_Major.wav')
rate, signal = wavfile.read('G_Major.wav')
G = Signal(rate, signal)
G.plot(True)


# In[126]:

def generate_chord():
    C = make_chord(['C','E','G'], 2, 'C_Major.wav')
    F = make_chord(['F','A','C'], 2, 'F_Major.wav')
    G = make_chord(['G','B','D'], 2, 'G_Major.wav')
    Am = make_chord(['A','C','E'], 2, 'A_Minor.wav')

    Am.write_to_file("chord1.wav")

    infiles = ["C_Major.wav", "F_Major.wav"]
    outfile = "chord2.wav"
    
    data= []
    for infile in infiles:
        w = wave.open(infile, 'rb')
        data.append( [w.getparams(), w.readframes(w.getnframes())] )
        w.close()

    output = wave.open(outfile, 'wb')
    output.setparams(data[0][0])
    output.writeframes(data[0][1])
    output.writeframes(data[1][1])
    
    output.close()
    
