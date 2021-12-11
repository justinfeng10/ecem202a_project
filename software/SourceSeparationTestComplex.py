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

# Calculate spectrogram and return the spectrogram
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=512, frame_step=128)
    return spectrogram
 
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
new_model = tf.keras.models.load_model('saved_model/my_model_VAWDComplex', compile=False)
        
# Read input file from folder
audio_binary = tf.io.read_file(filename)
waveform, sr = decode_audio(audio_binary)
spectrogram = get_spectrogram(waveform)

# Pass audio into ML model and obtain predicted outputs
mag = tf.abs(spectrogram)
phase = tf.math.angle(spectrogram)
complexVal = tf.stack([mag, phase], axis = 2)
complexVal = tf.expand_dims(complexVal, axis=0)
output = new_model(complexVal, training=True)
output0 = output[:, :, :, 0:2]
output1 = output[:, :, :, 2:4]
output2 = output[:, :, :, 4:6]
output3 = output[:, :, :, 6:8]
sum = tf.add(output0, output1)
sum = tf.add(sum, output2)
sum = tf.add(sum, output3)
filter0 = tf.math.divide_no_nan(output0, sum) 
filter1 = tf.math.divide_no_nan(output1, sum)
filter2 = tf.math.divide_no_nan(output2, sum) 
filter3 = tf.math.divide_no_nan(output3, sum)
predictedOutput0 = tf.multiply(filter0, complexVal)
predictedOutput1 = tf.multiply(filter1, complexVal)
predictedOutput2 = tf.multiply(filter2, complexVal)
predictedOutput3 = tf.multiply(filter3, complexVal)

# Calculate inverse stft for inputs as well as outputs, then save them to output file
inverse_stftx = tf.signal.inverse_stft(spectrogram, 512, 128)
inverse_stftx = tf.squeeze(inverse_stftx)
file = tf.audio.encode_wav(tf.expand_dims(inverse_stftx, 1), 16000, name=None)
tf.io.write_file("SourceSeparationTestOutput/VAWDComplex/original.wav", file, name=None)

magComplex = predictedOutput0[:, :, :, 0]
magComplex = tf.cast(magComplex, tf.complex64)
phaseComplex = predictedOutput0[:, :, :, 1]
phaseComplex = tf.cast(phaseComplex, tf.complex64)
spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
spectrogramx = tf.squeeze(spectrogramx)
inverse_stftx = tf.signal.inverse_stft(spectrogramx, 512, 128)
inverse_stftx = tf.squeeze(inverse_stftx)
file = tf.audio.encode_wav(tf.expand_dims(inverse_stftx, 1), 16000, name=None)
tf.io.write_file("SourceSeparationTestOutput/VAWDComplex/vacuumOutput.wav", file, name=None)

magComplex = predictedOutput1[:, :, :, 0]
magComplex = tf.cast(magComplex, tf.complex64)
phaseComplex = predictedOutput1[:, :, :, 1]
phaseComplex = tf.cast(phaseComplex, tf.complex64)
spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
spectrogramx = tf.squeeze(spectrogramx)
inverse_stftx = tf.signal.inverse_stft(spectrogramx, 512, 128)
inverse_stftx = tf.squeeze(inverse_stftx)
file = tf.audio.encode_wav(tf.expand_dims(inverse_stftx, 1), 16000, name=None)
tf.io.write_file("SourceSeparationTestOutput/VAWDComplex/alarmOutput.wav", file, name=None)

magComplex = predictedOutput2[:, :, :, 0]
magComplex = tf.cast(magComplex, tf.complex64)
phaseComplex = predictedOutput2[:, :, :, 1]
phaseComplex = tf.cast(phaseComplex, tf.complex64)
spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
spectrogramx = tf.squeeze(spectrogramx)
inverse_stftx = tf.signal.inverse_stft(spectrogramx, 512, 128)
inverse_stftx = tf.squeeze(inverse_stftx)
file = tf.audio.encode_wav(tf.expand_dims(inverse_stftx, 1), 16000, name=None)
tf.io.write_file("SourceSeparationTestOutput/VAWDComplex/waterOutput.wav", file, name=None)

magComplex = predictedOutput3[:, :, :, 0]
magComplex = tf.cast(magComplex, tf.complex64)
phaseComplex = predictedOutput3[:, :, :, 1]
phaseComplex = tf.cast(phaseComplex, tf.complex64)
spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
spectrogramx = tf.squeeze(spectrogramx)
inverse_stftx = tf.signal.inverse_stft(spectrogramx, 512, 128)
inverse_stftx = tf.squeeze(inverse_stftx)
file = tf.audio.encode_wav(tf.expand_dims(inverse_stftx, 1), 16000, name=None)
tf.io.write_file("SourceSeparationTestOutput/VAWDComplex/dogOutput.wav", file, name=None)

# Plot spectrograms
fig, axes = plt.subplots(1, 5, figsize=(40, 7))
ax = axes[0]
plot_spectrogram(np.squeeze(tf.abs(spectrogram).numpy()), ax)
ax.set_title("Mixture")
ax.axis('off')
ax = axes[1]
plot_spectrogram(np.squeeze(predictedOutput0[:, :, :, 0].numpy()), ax)
ax.set_title("Predicted Vacuum Cleaner")
ax.axis('off')
ax = axes[2]
plot_spectrogram(np.squeeze(predictedOutput1[:, :, :, 0].numpy()), ax)
ax.set_title("Predicted Alarm")
ax.axis('off')
ax = axes[3]
plot_spectrogram(np.squeeze(predictedOutput2[:, :, :, 0].numpy()), ax)
ax.set_title("Predicted Water")
ax.axis('off')
ax = axes[4]
plot_spectrogram(np.squeeze(predictedOutput3[:, :, :, 0].numpy()), ax)
ax.set_title("Predicted Dog")
ax.axis('off')
plt.show()
