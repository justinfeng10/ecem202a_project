# Imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
import sys

# Grab file name from command line
filename = sys.argv[1]
print(filename)

# Grab Wav file from folder
def decode_audio(audio_binary):
    audio, sr = tf.audio.decode_wav(audio_binary, desired_channels=1)
    return tf.squeeze(audio, axis=-1), sr

# Calculate spectrogram and return the magnitude and phase
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
        
# Read input file from folder
audio_binary = tf.io.read_file(filename)
waveform, sr = decode_audio(audio_binary)
spec, phase = get_spectrogram(waveform)
spec = tf.expand_dims(spec, -1)
phase = tf.expand_dims(phase, -1)

# Pass audio into ML model and obtain predicted outputs
audio = tf.expand_dims(spec, axis=0)
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

# Calculate inverse stft for inputs as well as outputs, then save them to output file
magComplex = tf.cast(audio, tf.complex64)
phaseComplex = tf.cast(phase, tf.complex64)
spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
spectrogramx = tf.squeeze(spectrogramx)
inverse_stftx = tf.signal.inverse_stft(spectrogramx, 512, 128)
inverse_stftx = tf.squeeze(inverse_stftx)
file = tf.audio.encode_wav(tf.expand_dims(inverse_stftx, 1), 16000, name=None)
tf.io.write_file("SourceSeparationTestOutput/VAWD/original.wav", file, name=None)

magComplex = tf.cast(predictedOutput0, tf.complex64)
phaseComplex = tf.cast(phase, tf.complex64)
spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
spectrogramx = tf.squeeze(spectrogramx)
inverse_stftx = tf.signal.inverse_stft(spectrogramx, 512, 128)
inverse_stftx = tf.squeeze(inverse_stftx)
file = tf.audio.encode_wav(tf.expand_dims(inverse_stftx, 1), 16000, name=None)
tf.io.write_file("SourceSeparationTestOutput/VAWD/vacuumOutput.wav", file, name=None)

magComplex = tf.cast(predictedOutput1, tf.complex64)
phaseComplex = tf.cast(phase, tf.complex64)
spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
spectrogramx = tf.squeeze(spectrogramx)
inverse_stftx = tf.signal.inverse_stft(spectrogramx, 512, 128)
inverse_stftx = tf.squeeze(inverse_stftx)
file = tf.audio.encode_wav(tf.expand_dims(inverse_stftx, 1), 16000, name=None)
tf.io.write_file("SourceSeparationTestOutput/VAWD/alarmOutput.wav", file, name=None)

magComplex = tf.cast(predictedOutput2, tf.complex64)
phaseComplex = tf.cast(phase, tf.complex64)
spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
spectrogramx = tf.squeeze(spectrogramx)
inverse_stftx = tf.signal.inverse_stft(spectrogramx, 512, 128)
inverse_stftx = tf.squeeze(inverse_stftx)
file = tf.audio.encode_wav(tf.expand_dims(inverse_stftx, 1), 16000, name=None)
tf.io.write_file("SourceSeparationTestOutput/VAWD/waterOutput.wav", file, name=None)

magComplex = tf.cast(predictedOutput3, tf.complex64)
phaseComplex = tf.cast(phase, tf.complex64)
spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
spectrogramx = tf.squeeze(spectrogramx)
inverse_stftx = tf.signal.inverse_stft(spectrogramx, 512, 128)
inverse_stftx = tf.squeeze(inverse_stftx)
file = tf.audio.encode_wav(tf.expand_dims(inverse_stftx, 1), 16000, name=None)
tf.io.write_file("SourceSeparationTestOutput/VAWD/dogOutput.wav", file, name=None)

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
plt.show()
