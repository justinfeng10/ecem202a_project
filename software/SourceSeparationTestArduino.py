# Imports
import wave
import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from scipy.io.wavfile import write

# Set up serial
arduino = serial.Serial(port='COM5', baudrate=9600, timeout=.1)
CHANNELS = 1

# Read from arduino
def write_read():
    time.sleep(0.01)
    data = arduino.readline()
    return data

# Name of created wav file
filename = "temp.wav"

# Grab Wav file from folder
def decode_audio(audio_binary):
    audio, sr = tf.audio.decode_wav(audio_binary, desired_channels=1)
    return tf.squeeze(audio, axis=-1), sr

# Calculate spectrogram and return the magnitude and phase
# Calculate spectrogram
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=512, frame_step=128)
    mag = tf.abs(spectrogram)
    phase = tf.math.angle(spectrogram)
    return mag, phase
  
# Plot spectrogram, given a spectrogram and an axis 
def plot_spectrogram(spectrogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns). An epsilon is added to avoid log of zero.
    log_spec = np.log(spectrogram.T+np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)
  
# Grab saved ML Model
new_model = tf.keras.models.load_model('saved_model/my_model_VAWD', compile=False)
  
# Infinite loop
while True:
    keyb = input("\n Press v to get 1 sec of audio data:")
    # Send data over serial if receive input from user
    if keyb == "v":
        arduino.write(b'v')
        frames = []
        # Read data from arduino
        while True:
            data = write_read()
            length = len(data)
            byteData = data[0:length-2]
            stringData = byteData.decode('ascii')
            if stringData == '':
                print("done!/n")
                break
            intData = int(stringData)
            frames.append(intData)
        # Write wav file to computer, for source separation code
        framesArray = np.array(frames)
        write("VA.wav",16000,framesArray.astype(np.int16))
        
        # Read file
        file = "VA.wav"
        audio_binary = tf.io.read_file(file)
        
        # Preprocess audio
        waveform, sr = decode_audio(audio_binary)
        spec, phase = get_spectrogram(waveform)
        spec = tf.expand_dims(spec, -1)
        phase = tf.expand_dims(phase, -1)
        audio = tf.expand_dims(spec, axis=0)
        
        # Pass audio into ML model and obtain predicted outputs
        output = new_model(audio)
        output0 = output[:, :, :, 0]
        output0 = tf.expand_dims(output0, -1)
        output1 = output[:, :, :, 1]
        output1 = tf.expand_dims(output1, -1)
        output2 = output[:, :, :, 2]
        output2 = tf.expand_dims(output2, -1)
        output3 = output[:, :, :, 3]
        output3 = tf.expand_dims(output3, -1)
        sum = tf.add(output0, output1)
        sum = tf.add(sum, output2)
        sum = tf.add(sum, output3)
        filter0 = tf.math.divide_no_nan(output0, sum) 
        filter1 = tf.math.divide_no_nan(output1, sum)
        filter2 = tf.math.divide_no_nan(output2, sum) 
        filter3 = tf.math.divide_no_nan(output3, sum)
        predictedOutput0 = tf.multiply(filter0, audio)
        predictedOutput1 = tf.multiply(filter1, audio)
        predictedOutput2 = tf.multiply(filter2, audio)
        predictedOutput3 = tf.multiply(filter3, audio)
        
        # Plot spectrograms
        fig, axes = plt.subplots(1, 5, figsize=(40, 7))
        ax = axes[0]
        plot_spectrogram(np.squeeze(audio.numpy()), ax)
        ax.set_title("Mixture")
        ax.axis('off')
        ax = axes[1]
        plot_spectrogram(np.squeeze(predictedOutput0.numpy()), ax)
        ax.set_title("Predicted Vacuum Cleaner")
        ax.axis('off')
        ax = axes[2]
        plot_spectrogram(np.squeeze(predictedOutput1.numpy()), ax)
        ax.set_title("Predicted Alarm")
        ax.axis('off')
        ax = axes[3]
        plot_spectrogram(np.squeeze(predictedOutput2.numpy()), ax)
        ax.set_title("Predicted Water")
        ax.axis('off')
        ax = axes[4]
        plot_spectrogram(np.squeeze(predictedOutput3.numpy()), ax)
        ax.set_title("Predicted Dog")
        ax.axis('off')
        
        # Calculate inverse stft for inputs as well as outputs, then save them to output file
        magComplex = tf.cast(predictedOutput0, tf.complex64)
        phaseComplex = tf.cast(phase, tf.complex64)
        spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
        spectrogramx = tf.squeeze(spectrogramx)
        inverse_stftx = tf.signal.inverse_stft(spectrogramx, 512, 128)
        inverse_stftx = tf.squeeze(inverse_stftx)
        file = tf.audio.encode_wav(tf.expand_dims(inverse_stftx, 1), 16000, name=None)
	tf.io.write_file("SourceSeparationTestOutput/Arduino/vacuumOutput.wav", file, name=None)
        
        magComplex = tf.cast(predictedOutput1, tf.complex64)
        phaseComplex = tf.cast(phase, tf.complex64)
        spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
        spectrogramx = tf.squeeze(spectrogramx)
        inverse_stftx = tf.signal.inverse_stft(spectrogramx, 512, 128)
        inverse_stftx = tf.squeeze(inverse_stftx)
        file = tf.audio.encode_wav(tf.expand_dims(inverse_stftx, 1), 16000, name=None)
	tf.io.write_file("SourceSeparationTestOutput/Arduino/alarmOutput.wav", file, name=None)
        
        magComplex = tf.cast(predictedOutput2, tf.complex64)
        phaseComplex = tf.cast(phase, tf.complex64)
        spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
        spectrogramx = tf.squeeze(spectrogramx)
        inverse_stftx = tf.signal.inverse_stft(spectrogramx, 512, 128)
        inverse_stftx = tf.squeeze(inverse_stftx)
        file = tf.audio.encode_wav(tf.expand_dims(inverse_stftx, 1), 16000, name=None)
	tf.io.write_file("SourceSeparationTestOutput/Arduino/waterOutput.wav", file, name=None)
        
        magComplex = tf.cast(predictedOutput3, tf.complex64)
        phaseComplex = tf.cast(phase, tf.complex64)
        spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
        spectrogramx = tf.squeeze(spectrogramx)
        inverse_stftx = tf.signal.inverse_stft(spectrogramx, 512, 128)
        inverse_stftx = tf.squeeze(inverse_stftx)
        file = tf.audio.encode_wav(tf.expand_dims(inverse_stftx, 1), 16000, name=None)
	tf.io.write_file("SourceSeparationTestOutput/Arduino/dogOutput.wav", file, name=None)
    
