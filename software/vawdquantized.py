# Imports
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import tensorflow_io as tfio
from IPython import display
from scipy.io import wavfile
import museval
import json
import glob
import math
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import layers
from tensorflow.keras import models

# Grab Wav file from folder
# Preprocessing code here and below modified from
# Tensorflow Example: Simple Audio Recognition
# Link: https://www.tensorflow.org/tutorials/audio/simple_audio
def decode_audio(audio_binary):
    audio, sr = tf.audio.decode_wav(audio_binary, desired_channels=1)
    return tf.squeeze(audio, axis=-1), sr

# For a given filepath ending with a wav file, obtain the waveform and the label
# Function for Vacuum
def get_waveform_and_label_vacuum(file_path):
    label = "Vacuum Cleaner"
    audio_binary = tf.io.read_file(file_path)
    waveform, sr = decode_audio(audio_binary)
    if (sr == 48000):
      waveform = tfio.audio.resample(waveform, 48000, 16000, name=None)
    else:
      waveform = tfio.audio.resample(waveform, 44100, 16000, name=None)
    zeros = tf.zeros([16000,])
    return waveform, waveform, zeros, zeros, zeros, label

# For a given filepath ending with a wav file, obtain the waveform and the label
# Function for Alarm
def get_waveform_and_label_alarm(file_path):
    label = "Alarm"
    audio_binary = tf.io.read_file(file_path)
    waveform, sr = decode_audio(audio_binary)
    if (sr == 48000):
      waveform = tfio.audio.resample(waveform, 48000, 16000, name=None)
    else:
      waveform = tfio.audio.resample(waveform, 44100, 16000, name=None)
    zeros = tf.zeros([16000,])
    return waveform, zeros, waveform, zeros, zeros, label

# For a given filepath ending with a wav file, obtain the waveform and the label
# Function for Water
def get_waveform_and_label_water(file_path):
    label = "Water"
    audio_binary = tf.io.read_file(file_path)
    waveform, sr = decode_audio(audio_binary)
    waveform = tfio.audio.resample(waveform, 44100, 16000, name=None)
    zeros = tf.zeros([16000,])
    return waveform, zeros, zeros, waveform, zeros, label

# For a given filepath ending with a wav file, obtain the waveform and the label
# Function for Dog
def get_waveform_and_label_dog(file_path):
    label = "Dog"
    audio_binary = tf.io.read_file(file_path)
    waveform, sr = decode_audio(audio_binary)
    if (sr == 48000):
      waveform = tfio.audio.resample(waveform, 48000, 16000, name=None)
    else:
      waveform = tfio.audio.resample(waveform, 44100, 16000, name=None)
    zeros = tf.zeros([16000,])
    return waveform, zeros, zeros, zeros, waveform, label

# For a given filepath ending with a wav file, obtain the waveform and the label
# Function for MIXTURE
def get_waveform_and_label_mixture(file_path):
    label = "Mixture"
    audio_binary = tf.io.read_file(file_path)
    waveform, sr = decode_audio(audio_binary)
    if (sr == 48000):
      waveform = tfio.audio.resample(waveform, 48000, 16000, name=None)
    else:
      waveform = tfio.audio.resample(waveform, 44100, 16000, name=None)
    
    # Obtain original files for vacuum and pouring
    audio_binary = tf.io.read_file(file_path + "Label/0.wav")
    vacuumCleanerLabel, sr = decode_audio(audio_binary)
    if (sr == 48000):
      vacuumCleanerLabel = tfio.audio.resample(vacuumCleanerLabel, 48000, 16000, name=None)
    else:
      vacuumCleanerLabel = tfio.audio.resample(vacuumCleanerLabel, 44100, 16000, name=None)
    audio_binary = tf.io.read_file(file_path + "Label/1.wav")
    alarmLabel, sr = decode_audio(audio_binary)
    if (sr == 48000):
      alarmLabel = tfio.audio.resample(alarmLabel, 48000, 16000, name=None)
    else:
      alarmLabel = tfio.audio.resample(alarmLabel, 44100, 16000, name=None)
    audio_binary = tf.io.read_file(file_path + "Label/2.wav")
    waterLabel, sr = decode_audio(audio_binary)
    if (sr == 48000):
      waterLabel = tfio.audio.resample(waterLabel, 48000, 16000, name=None)
    else:
      waterLabel = tfio.audio.resample(waterLabel, 44100, 16000, name=None)
    audio_binary = tf.io.read_file(file_path + "Label/3.wav")
    dogLabel, sr = decode_audio(audio_binary)
    if (sr == 48000):
      dogLabel = tfio.audio.resample(dogLabel, 48000, 16000, name=None)
    else:
      dogLabel = tfio.audio.resample(dogLabel, 44100, 16000, name=None)

    return waveform, vacuumCleanerLabel, alarmLabel, waterLabel, dogLabel, label

# Dataset: Vacuum
data_dir = pathlib.Path('Sounds/Esc/VacuumCleanerCut')
filenames = tf.io.gfile.glob(str(data_dir) + '/*')
filenames = tf.random.shuffle(filenames)
AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(filenames)
waveformVacuum_ds = files_ds.map(get_waveform_and_label_vacuum, num_parallel_calls=AUTOTUNE)

# Dataset: Alarm
data_dir = pathlib.Path('Sounds/Esc/AlarmCut')
filenames = tf.io.gfile.glob(str(data_dir) + '/*')
filenames = tf.random.shuffle(filenames)
AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(filenames)
waveformAlarm_ds = files_ds.map(get_waveform_and_label_alarm, num_parallel_calls=AUTOTUNE)

# Dataset: Water
data_dir = pathlib.Path('Sounds/Esc/PouringWaterCut')
filenames = tf.io.gfile.glob(str(data_dir) + '/*')
filenames = tf.random.shuffle(filenames)
AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(filenames)
waveformWater_ds = files_ds.map(get_waveform_and_label_water, num_parallel_calls=AUTOTUNE)

# Dataset: Dog
data_dir = pathlib.Path('Sounds/Esc/DogCut')
filenames = tf.io.gfile.glob(str(data_dir) + '/*')
filenames = tf.random.shuffle(filenames)
AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(filenames)
waveformDog_ds = files_ds.map(get_waveform_and_label_dog, num_parallel_calls=AUTOTUNE)

# Dataset: Mixtures
data_dir = pathlib.Path('Sounds/Esc/MixturesVAWD')
filenames = tf.io.gfile.glob(str(data_dir) + '/*1.wav')
filenames = tf.random.shuffle(filenames)
AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(filenames)
waveformMixture_ds = files_ds.map(get_waveform_and_label_mixture, num_parallel_calls=AUTOTUNE)

# Add the mixture structure to the existing structure of singular wav files
waveform_esc_ds = waveformVacuum_ds.concatenate(waveformAlarm_ds)
waveform_esc_ds = waveform_esc_ds.concatenate(waveformWater_ds)
waveform_esc_ds = waveform_esc_ds.concatenate(waveformDog_ds)
waveform_esc_ds = waveform_esc_ds.concatenate(waveformMixture_ds)

# Dataset: Vacuum
data_dir = pathlib.Path('Sounds/Desed/Vacuum_cleanerCut')
filenames = tf.io.gfile.glob(str(data_dir) + '/*')
filenames = tf.random.shuffle(filenames)
AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(filenames)
waveformVacuum_ds = files_ds.map(get_waveform_and_label_vacuum, num_parallel_calls=AUTOTUNE)

# Dataset: Alarm
data_dir = pathlib.Path('Sounds/Desed/Alarm_bell_ringingCut')
filenames = tf.io.gfile.glob(str(data_dir) + '/*')
filenames = tf.random.shuffle(filenames)
AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(filenames)
waveformAlarm_ds = files_ds.map(get_waveform_and_label_alarm, num_parallel_calls=AUTOTUNE)

# Dataset: Water
data_dir = pathlib.Path('Sounds/Desed/Running_waterCut')
filenames = tf.io.gfile.glob(str(data_dir) + '/*')
filenames = tf.random.shuffle(filenames)
AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(filenames)
waveformWater_ds = files_ds.map(get_waveform_and_label_water, num_parallel_calls=AUTOTUNE)

# Dataset: Dog
data_dir = pathlib.Path('Sounds/Desed/DogCut')
filenames = tf.io.gfile.glob(str(data_dir) + '/*')
filenames = tf.random.shuffle(filenames)
AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(filenames)
waveformDog_ds = files_ds.map(get_waveform_and_label_dog, num_parallel_calls=AUTOTUNE)

# Dataset: Mixtures
data_dir = pathlib.Path('Sounds/Desed/MixturesVAWD')
filenames = tf.io.gfile.glob(str(data_dir) + '/*1.wav')
filenames = tf.random.shuffle(filenames)
AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(filenames)
waveformMixture_ds = files_ds.map(get_waveform_and_label_mixture, num_parallel_calls=AUTOTUNE)

# Add the mixture structure to the existing structure of singular wav files, then mix up wav files
waveformVacuum_ds = waveformVacuum_ds.shuffle(345, reshuffle_each_iteration=False).take(100)
waveformAlarm_ds = waveformAlarm_ds.shuffle(290, reshuffle_each_iteration=False).take(100)
waveformWater_ds = waveformWater_ds.shuffle(315, reshuffle_each_iteration=False).take(100)
waveformDog_ds = waveformDog_ds.shuffle(265, reshuffle_each_iteration=False).take(100)
waveformMixture_ds = waveformMixture_ds.shuffle(1200, reshuffle_each_iteration=False).take(400)
waveform_desed_ds = waveformVacuum_ds.concatenate(waveformAlarm_ds)
waveform_desed_ds = waveform_desed_ds.concatenate(waveformWater_ds)
waveform_desed_ds = waveform_desed_ds.concatenate(waveformDog_ds)
waveform_desed_ds = waveform_desed_ds.concatenate(waveformMixture_ds)

# Concatenate ESC and Desed datasets, then shuffle
waveform_ds = waveform_esc_ds.concatenate(waveform_desed_ds)
waveform_ds = waveform_ds.shuffle(2400, reshuffle_each_iteration=False)

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

# Given the raw input (audio), and the components that make up
# the input (label0, label1, label2, label3), and the actual label, compute
# the spectrograms and return the phase of the raw input
def get_spectrogram_and_label_id(audio, label0, label1, label2, label3, label):
    spectrogram, phase = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    phase = tf.expand_dims(phase, -1)
    spectrogram0, phase0 = get_spectrogram(label0)
    spectrogram1, phase1 = get_spectrogram(label1)
    spectrogram2, phase2 = get_spectrogram(label2)
    spectrogram3, phase3 = get_spectrogram(label3)
    label_id = label
    return spectrogram, phase, spectrogram0, spectrogram1, spectrogram2, spectrogram3, label_id

# Dataset: spectrograms
spectrogram_ds = waveform_ds.map(
    get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

# Define train, validation, test datasets
train_size = int(0.8*2400)
val_size = int(0.1*2400)
train_ds = spectrogram_ds.take(train_size)    
val_ds = spectrogram_ds.skip(train_size).take(val_size)
test_ds = spectrogram_ds.skip(train_size).skip(val_size)

# Batch the training input
batch_size = 2
train_ds = train_ds.batch(batch_size)

# Convert to fixed point
@tf.custom_gradient
def to_fixedpoint(value, wl=8, fl=4):
    # Find max number of values supported
    max = 2**wl
    # Find max and min numbers (scaled up numbers)
    maxPosInt = (max / 2) - 1
    maxNegInt = -(max / 2)
    # Shift values to the left
    value_scaled = value * (2**fl)
    # Cut off unsupported bits
    value_int = tf.cast(value_scaled, tf.int32)
    # Convert back to float32 format
    value_double = tf.cast(value_int, tf.float32)
    # Set number to maximum if >= maximum.
    # Set number to mininmum if <= minimum.
    comparison = tf.greater_equal(value_double, tf.constant(maxPosInt))   
    value = tf.where(comparison, tf.constant(maxPosInt), value_double)
    comparison = tf.less_equal(value_double, tf.constant(maxNegInt))    
    value = tf.where(comparison, tf.constant(maxNegInt), value_double)
    # Shift back to the right
    value_double = value / (2**fl)
    value_q = value_double

    def grad(upstream):
        return upstream,0.0,0.0

    return value_q, grad

# Source Separation ML Model
class SourceSeparationModel(tf.keras.Model):
    def __init__(self):
        super(SourceSeparationModel, self).__init__()
        # Encoder
        self.conv1 = Conv2D(filters=30, kernel_size=(1, 257))
        self.conv2 = Conv2D(filters=15, kernel_size=(30, 1))
        self.d1 = Dense(units=128, activation='relu')
        # Decoder
        self.dClass0 = Dense(units=15, activation='relu')
        self.dClass1 = Dense(units=15, activation='relu')
        self.dClass2 = Dense(units=15, activation='relu')
        self.dClass3 = Dense(units=15, activation='relu')
        self.conv3Class0 = Conv2DTranspose(filters=15, kernel_size=(30, 1))
        self.conv4Class0 = Conv2DTranspose(filters=30, kernel_size=(1, 257))
        self.conv3Class1 = Conv2DTranspose(filters=15, kernel_size=(30, 1))
        self.conv4Class1 = Conv2DTranspose(filters=30, kernel_size=(1, 257))
        self.conv3Class2 = Conv2DTranspose(filters=15, kernel_size=(30, 1))
        self.conv4Class2 = Conv2DTranspose(filters=30, kernel_size=(1, 257))
        self.conv3Class3 = Conv2DTranspose(filters=15, kernel_size=(30, 1))
        self.conv4Class3 = Conv2DTranspose(filters=30, kernel_size=(1, 257))
        self.concat = Concatenate()
        # Output
        self.out = Dense(units=4, activation='relu')

    def call(self, x):
        x = to_fixedpoint(x, 8, 4)
        x = self.conv1(x)
        x = to_fixedpoint(x, 8, 4)
        x = self.conv2(x)
        x = to_fixedpoint(x, 8, 4)
        x = self.d1(x)
        x = to_fixedpoint(x, 8, 4)
        out0 = self.dClass0(x)
        out0 = to_fixedpoint(out0, 8, 4)
        out0 = self.conv3Class0(out0)
        out0 = to_fixedpoint(out0, 8, 4)
        out0 = self.conv4Class0(out0)
        out1 = self.dClass1(x)
        out1 = to_fixedpoint(out1, 8, 4)
        out1 = self.conv3Class1(out1)
        out1 = to_fixedpoint(out1, 8, 4)
        out1 = self.conv4Class1(out1)
        out2 = self.dClass2(x)
        out2 = to_fixedpoint(out2, 8, 4)
        out2 = self.conv3Class2(out2)
        out2 = to_fixedpoint(out2, 8, 4)
        out2 = self.conv4Class2(out2)
        out3 = self.dClass3(x)
        out3 = to_fixedpoint(out3, 8, 4)
        out3 = self.conv3Class3(out3)
        out3 = to_fixedpoint(out3, 8, 4)
        out3 = self.conv4Class3(out3)
        output = self.concat([out0, out1, out2, out3])
        output = to_fixedpoint(output, 8, 4)
        return self.out(output)

# Train Function
def train(model, optimizer, epochs):
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    # Parameter for loss function
    alpha = 0.001
    for epoch in range(epochs):  
        train_loss.reset_states()
        for x_np, phase, spectrogram0, spectrogram1, spectrogram2, spectrogram3, label in train_ds:
            with tf.GradientTape() as tape:
                output = model(x_np, training=True)
                # Isolate outputs
                output0 = output[:, :, :, 0]
                output0 = tf.expand_dims(output0, -1)
                output1 = output[:, :, :, 1]
                output1 = tf.expand_dims(output1, -1)
                output2 = output[:, :, :, 2]
                output2 = tf.expand_dims(output2, -1)
                output3 = output[:, :, :, 3]
                output3 = tf.expand_dims(output3, -1)
                
                # Compute time frequency mask
                sum = tf.add(output0, output1)
                sum = tf.add(sum, output2)
                sum = tf.add(sum, output3)
                filter0 = tf.math.divide_no_nan(output0, sum) 
                filter1 = tf.math.divide_no_nan(output1, sum)
                filter2 = tf.math.divide_no_nan(output2, sum) 
                filter3 = tf.math.divide_no_nan(output3, sum)
                
                # Find predicted outputs
                predictedOutput0 = tf.multiply(filter0, x_np)
                predictedOutput1 = tf.multiply(filter1, x_np)
                predictedOutput2 = tf.multiply(filter2, x_np)
                predictedOutput3 = tf.multiply(filter3, x_np)
                predictedOutput0 = tf.squeeze(predictedOutput0, axis=3)
                predictedOutput1 = tf.squeeze(predictedOutput1, axis=3)
                predictedOutput2 = tf.squeeze(predictedOutput2, axis=3)
                predictedOutput3 = tf.squeeze(predictedOutput3, axis=3)

                # Calculate Loss
                mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
                loss0 = mse(predictedOutput0, spectrogram0)
                loss1 = mse(predictedOutput1, spectrogram1)
                loss2 = mse(predictedOutput2, spectrogram2)
                loss3 = mse(predictedOutput3, spectrogram3)
                alpha0 = alpha * mse(predictedOutput0, predictedOutput1) + alpha * mse(predictedOutput0, predictedOutput2) + alpha * mse(predictedOutput0, predictedOutput3)
                alpha1 = alpha * mse(predictedOutput1, predictedOutput0) + alpha * mse(predictedOutput1, predictedOutput2) + alpha * mse(predictedOutput1, predictedOutput3)
                alpha2 = alpha * mse(predictedOutput2, predictedOutput0) + alpha * mse(predictedOutput2, predictedOutput1) + alpha * mse(predictedOutput2, predictedOutput3)
                alpha3 = alpha * mse(predictedOutput3, predictedOutput0) + alpha * mse(predictedOutput3, predictedOutput1) + alpha * mse(predictedOutput3, predictedOutput2)
                lossTemp0 = tf.add(loss0, loss1)
                lossTemp0 = tf.add(lossTemp0, loss2)
                lossTemp0 = tf.add(lossTemp0, loss3)
                lossTemp0 = tf.abs(lossTemp0)
                lossTemp1 = tf.add(alpha0, alpha1)
                lossTemp1 = tf.add(lossTemp1, alpha2)
                lossTemp1 = tf.add(lossTemp1, alpha3)
                lossTemp1 = tf.abs(lossTemp1)
                # Subtract out differences between predicted outputs
                loss = tf.abs(tf.subtract(lossTemp0, lossTemp1))
                # Use original term
                loss = lossTemp0

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                train_loss.update_state(loss)

        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch+1, train_loss.result()))

plt.show()

# Instantiate model
model = SourceSeparationModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, name = "Adam")
train(model, optimizer, epochs=20)
print(model.summary())

# Initialize Metrics for different sources
vcCSDR = 0
vcCSIR = 0
vcCSAR = 0
aCSDR = 0
aCSIR = 0
aCSAR = 0
wCSDR = 0
wCSIR = 0
wCSAR = 0
dCSDR = 0
dCSIR = 0
dCSAR = 0
sdrV = 0
sdrA = 0
sdrW = 0
sdrD = 0
sirV = 0
sirA = 0
sirW = 0
sirD = 0
sarV = 0
sarA = 0
sarW = 0
sarD = 0
count = 0
for audio, phase, sound0, sound1, sound2, sound3, label in test_ds:
    # Pass audio into ML model and obtain predicted outputs
    audio = tf.expand_dims(audio, axis=0)
    output = model(audio)
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
    sound0 = tf.expand_dims(sound0, 0)
    sound0 = tf.expand_dims(sound0, 3)
    sound1 = tf.expand_dims(sound1, 0)
    sound1 = tf.expand_dims(sound1, 3)
    sound2 = tf.expand_dims(sound2, 0)
    sound2 = tf.expand_dims(sound2, 3)
    sound3 = tf.expand_dims(sound3, 0)
    sound3 = tf.expand_dims(sound3, 3)

    # Calculate inverse stft for inputs as well as outputs
    magComplex = tf.cast(sound0, tf.complex64)
    phaseComplex = tf.cast(phase, tf.complex64)
    spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
    spectrogramx = tf.squeeze(spectrogramx)
    inverse_stftx0 = tf.signal.inverse_stft(spectrogramx, 512, 128)
    inverse_stftx0 = tf.squeeze(inverse_stftx0)

    magComplex = tf.cast(predictedOutput0, tf.complex64)
    phaseComplex = tf.cast(phase, tf.complex64)
    spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
    spectrogramx = tf.squeeze(spectrogramx)
    inverse_stftx1 = tf.signal.inverse_stft(spectrogramx, 512, 128)
    inverse_stftx1 = tf.squeeze(inverse_stftx1)

    magComplex = tf.cast(sound1, tf.complex64)
    phaseComplex = tf.cast(phase, tf.complex64)
    spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
    spectrogramx = tf.squeeze(spectrogramx)
    inverse_stftx2 = tf.signal.inverse_stft(spectrogramx, 512, 128)
    inverse_stftx2 = tf.squeeze(inverse_stftx2)

    magComplex = tf.cast(predictedOutput1, tf.complex64)
    phaseComplex = tf.cast(phase, tf.complex64)
    spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
    spectrogramx = tf.squeeze(spectrogramx)
    inverse_stftx3 = tf.signal.inverse_stft(spectrogramx, 512, 128)
    inverse_stftx3 = tf.squeeze(inverse_stftx3)

    magComplex = tf.cast(sound2, tf.complex64)
    phaseComplex = tf.cast(phase, tf.complex64)
    spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
    spectrogramx = tf.squeeze(spectrogramx)
    inverse_stftx4 = tf.signal.inverse_stft(spectrogramx, 512, 128)
    inverse_stftx4 = tf.squeeze(inverse_stftx4)

    magComplex = tf.cast(predictedOutput2, tf.complex64)
    phaseComplex = tf.cast(phase, tf.complex64)
    spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
    spectrogramx = tf.squeeze(spectrogramx)
    inverse_stftx5 = tf.signal.inverse_stft(spectrogramx, 512, 128)
    inverse_stftx5 = tf.squeeze(inverse_stftx5)

    magComplex = tf.cast(sound3, tf.complex64)
    phaseComplex = tf.cast(phase, tf.complex64)
    spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
    spectrogramx = tf.squeeze(spectrogramx)
    inverse_stftx6 = tf.signal.inverse_stft(spectrogramx, 512, 128)
    inverse_stftx6 = tf.squeeze(inverse_stftx6)

    magComplex = tf.cast(predictedOutput3, tf.complex64)
    phaseComplex = tf.cast(phase, tf.complex64)
    spectrogramx = magComplex * tf.math.exp(1j * phaseComplex)
    spectrogramx = tf.squeeze(spectrogramx)
    inverse_stftx7 = tf.signal.inverse_stft(spectrogramx, 512, 128)
    inverse_stftx7 = tf.squeeze(inverse_stftx7)

    # Prepare to process data
    string = str(label.numpy())
    length = len(string) 
    # Calculate metrics for Mixtures
    if (string[2:length-1] == "Mixture"):
        # Save inverse stfts into "References" and "Estimates" folders
        file0 = tf.audio.encode_wav(tf.expand_dims(inverse_stftx0, 1), 16000, name=None)
        tf.io.write_file("References/file0.wav", file0, name=None)
        file1 = tf.audio.encode_wav(tf.expand_dims(inverse_stftx1, 1), 16000, name=None)
        tf.io.write_file("Estimates/file0.wav", file1, name=None)

        file2 = tf.audio.encode_wav(tf.expand_dims(inverse_stftx2, 1), 16000, name=None)
        tf.io.write_file("References/file1.wav", file2, name=None)
        file3 = tf.audio.encode_wav(tf.expand_dims(inverse_stftx3, 1), 16000, name=None)
        tf.io.write_file("Estimates/file1.wav", file3, name=None)

        file4 = tf.audio.encode_wav(tf.expand_dims(inverse_stftx4, 1), 16000, name=None)
        tf.io.write_file("References/file2.wav", file4, name=None)
        file5 = tf.audio.encode_wav(tf.expand_dims(inverse_stftx5, 1), 16000, name=None)
        tf.io.write_file("Estimates/file2.wav", file5, name=None)

        file6 = tf.audio.encode_wav(tf.expand_dims(inverse_stftx6, 1), 16000, name=None)
        tf.io.write_file("References/file3.wav", file6, name=None)
        file7 = tf.audio.encode_wav(tf.expand_dims(inverse_stftx7, 1), 16000, name=None)
        tf.io.write_file("Estimates/file3.wav", file7, name=None)

        # Read in saved inverse stfts
        fs, ref0 = wavfile.read("References/file0.wav")
        fs, est0 = wavfile.read("Estimates/file0.wav")
        fs, ref1 = wavfile.read("References/file1.wav")
        fs, est1 = wavfile.read("Estimates/file1.wav")
        fs, ref2 = wavfile.read("References/file2.wav")
        fs, est2 = wavfile.read("Estimates/file2.wav")
        fs, ref3 = wavfile.read("References/file3.wav")
        fs, est3 = wavfile.read("Estimates/file3.wav")
        
        # Vacuum Alarm
        # Error checking (code utilized from Museval)
        if (not ((np.any(np.all(np.sum(ref0, axis=tuple(range(2, ref0.ndim))) == 0, axis=0))
           or np.any(np.all(np.sum(est0, axis=tuple(range(2, est0.ndim))) == 0, axis=0)))
           or (np.any(np.all(np.sum(ref1, axis=tuple(range(2, ref1.ndim))) == 0, axis=0))
           or np.any(np.all(np.sum(est1, axis=tuple(range(2, est1.ndim))) == 0, axis=0))))):
            
            os.remove("References/file2.wav")
            os.remove("Estimates/file2.wav")
            os.remove("References/file3.wav")
            os.remove("Estimates/file3.wav")
            
            # Calculate scores, save in "test.json"
            scores = museval.eval_dir("References", "Estimates", "Outputs")
            scores.save("test.json")

            # compute the metrics
            f = open('test.json',)

            # returns JSON object as
            # a dictionary
            data = json.load(f)

            # Iterating through the json
            # list
            # Metrics: Vacuum
            jsonData = data['targets'][1]['frames'][0]['metrics']
            sdrTemp = jsonData.get('SDR')
            sirTemp = jsonData.get('SIR')
            sarTemp = jsonData.get('SAR')
            if (not (math.isnan(sdrTemp))):
                sdrV = sdrV + sdrTemp
                vcCSDR = vcCSDR + 1
            if (not (math.isnan(sirTemp))):
                sirV = sirV + sirTemp
                vcCSIR = vcCSIR + 1
            if (not (math.isnan(sarTemp))):
                sarV = sarV + sarTemp
                vcCSAR = vcCSAR + 1

            # Metrics: Alarm
            jsonData = data['targets'][0]['frames'][0]['metrics']
            sdrTemp = jsonData.get('SDR')
            sirTemp = jsonData.get('SIR')
            sarTemp = jsonData.get('SAR')
            if (not (math.isnan(sdrTemp))):
                sdrA = sdrA + sdrTemp
                aCSDR = aCSDR + 1
            if (not (math.isnan(sirTemp))):
                sirA = sirA + sirTemp
                aCSIR = aCSIR + 1
            if (not (math.isnan(sarTemp))):
                sarA = sarA + sarTemp
                aCSAR = aCSAR + 1
        
        # Vacuum Water
        # Error checking
        if (not ((np.any(np.all(np.sum(ref0, axis=tuple(range(2, ref0.ndim))) == 0, axis=0))
           or np.any(np.all(np.sum(est0, axis=tuple(range(2, est0.ndim))) == 0, axis=0)))
           or (np.any(np.all(np.sum(ref2, axis=tuple(range(2, ref2.ndim))) == 0, axis=0))
           or np.any(np.all(np.sum(est2, axis=tuple(range(2, est2.ndim))) == 0, axis=0))))):
            
            os.remove("References/file1.wav")
            os.remove("Estimates/file1.wav")
            os.remove("References/file3.wav")
            os.remove("Estimates/file3.wav")
            
            # Calculate scores, save in "test.json"
            scores = museval.eval_dir("References", "Estimates", "Outputs")
            scores.save("test.json")

            # compute the metrics
            f = open('test.json',)

            # returns JSON object as
            # a dictionary
            data = json.load(f)

            # Iterating through the json
            # list
            # Metrics: Vacuum
            jsonData = data['targets'][1]['frames'][0]['metrics']
            sdrTemp = jsonData.get('SDR')
            sirTemp = jsonData.get('SIR')
            sarTemp = jsonData.get('SAR')
            if (not (math.isnan(sdrTemp))):
                sdrV = sdrV + sdrTemp
                vcCSDR = vcCSDR + 1
            if (not (math.isnan(sirTemp))):
                sirV = sirV + sirTemp
                vcCSIR = vcCSIR + 1
            if (not (math.isnan(sarTemp))):
                sarV = sarV + sarTemp
                vcCSAR = vcCSAR + 1

            # Metrics: Water
            jsonData = data['targets'][0]['frames'][0]['metrics']
            sdrTemp = jsonData.get('SDR')
            sirTemp = jsonData.get('SIR')
            sarTemp = jsonData.get('SAR')
            if (not (math.isnan(sdrTemp))):
                sdrW = sdrW + sdrTemp
                wCSDR = wCSDR + 1
            if (not (math.isnan(sirTemp))):
                sirW = sirW + sirTemp
                wCSIR = wCSIR + 1
            if (not (math.isnan(sarTemp))):
                sarW = sarW + sarTemp
                wCSAR = wCSAR + 1
        
        # Vacuum Dog
        # Error checking
        if (not ((np.any(np.all(np.sum(ref0, axis=tuple(range(2, ref0.ndim))) == 0, axis=0))
           or np.any(np.all(np.sum(est0, axis=tuple(range(2, est0.ndim))) == 0, axis=0)))
           or (np.any(np.all(np.sum(ref3, axis=tuple(range(2, ref3.ndim))) == 0, axis=0))
           or np.any(np.all(np.sum(est3, axis=tuple(range(2, est3.ndim))) == 0, axis=0))))):
            
            os.remove("References/file1.wav")
            os.remove("Estimates/file1.wav")
            os.remove("References/file2.wav")
            os.remove("Estimates/file2.wav")
            
            # Calculate scores, save in "test.json"
            scores = museval.eval_dir("References", "Estimates", "Outputs")
            scores.save("test.json")

            # compute the metrics
            f = open('test.json',)

            # returns JSON object as
            # a dictionary
            data = json.load(f)

            # Iterating through the json
            # list
            # Metrics: Vacuum
            jsonData = data['targets'][1]['frames'][0]['metrics']
            sdrTemp = jsonData.get('SDR')
            sirTemp = jsonData.get('SIR')
            sarTemp = jsonData.get('SAR')
            if (not (math.isnan(sdrTemp))):
                sdrV = sdrV + sdrTemp
                vcCSDR = vcCSDR + 1
            if (not (math.isnan(sirTemp))):
                sirV = sirV + sirTemp
                vcCSIR = vcCSIR + 1
            if (not (math.isnan(sarTemp))):
                sarV = sarV + sarTemp
                vcCSAR = vcCSAR + 1

            # Metrics: Dog
            jsonData = data['targets'][0]['frames'][0]['metrics']
            sdrTemp = jsonData.get('SDR')
            sirTemp = jsonData.get('SIR')
            sarTemp = jsonData.get('SAR')
            if (not (math.isnan(sdrTemp))):
                sdrD = sdrD + sdrTemp
                dCSDR = dCSDR + 1
            if (not (math.isnan(sirTemp))):
                sirD = sirD + sirTemp
                dCSIR = dCSIR + 1
            if (not (math.isnan(sarTemp))):
                sarD = sarD + sarTemp
                dCSAR = dCSAR + 1
        
        # Alarm Water
        # Error checking
        if (not ((np.any(np.all(np.sum(ref1, axis=tuple(range(2, ref1.ndim))) == 0, axis=0))
           or np.any(np.all(np.sum(est1, axis=tuple(range(2, est1.ndim))) == 0, axis=0)))
           or (np.any(np.all(np.sum(ref2, axis=tuple(range(2, ref2.ndim))) == 0, axis=0))
           or np.any(np.all(np.sum(est2, axis=tuple(range(2, est2.ndim))) == 0, axis=0))))):
            
            os.remove("References/file0.wav")
            os.remove("Estimates/file0.wav")
            os.remove("References/file3.wav")
            os.remove("Estimates/file3.wav")
            
            # Calculate scores, save in "test.json"
            scores = museval.eval_dir("References", "Estimates", "Outputs")
            scores.save("test.json")

            # compute the metrics
            f = open('test.json',)

            # returns JSON object as
            # a dictionary
            data = json.load(f)

            # Iterating through the json
            # list
            # Metrics: Alarm
            jsonData = data['targets'][1]['frames'][0]['metrics']
            sdrTemp = jsonData.get('SDR')
            sirTemp = jsonData.get('SIR')
            sarTemp = jsonData.get('SAR')
            if (not (math.isnan(sdrTemp))):
                sdrA = sdrA + sdrTemp
                aCSDR = aCSDR + 1
            if (not (math.isnan(sirTemp))):
                sirA = sirA + sirTemp
                aCSIR = aCSIR + 1
            if (not (math.isnan(sarTemp))):
                sarA = sarA + sarTemp
                aCSAR = aCSAR + 1

            # Metrics: Water
            jsonData = data['targets'][0]['frames'][0]['metrics']
            sdrTemp = jsonData.get('SDR')
            sirTemp = jsonData.get('SIR')
            sarTemp = jsonData.get('SAR')
            if (not (math.isnan(sdrTemp))):
                sdrW = sdrW + sdrTemp
                wCSDR = wCSDR + 1
            if (not (math.isnan(sirTemp))):
                sirW = sirW + sirTemp
                wCSIR = wCSIR + 1
            if (not (math.isnan(sarTemp))):
                sarW = sarW + sarTemp
                wCSAR = wCSAR + 1
        
        # Alarm Dog
        # Error checking
        if (not ((np.any(np.all(np.sum(ref1, axis=tuple(range(2, ref1.ndim))) == 0, axis=0))
           or np.any(np.all(np.sum(est1, axis=tuple(range(2, est1.ndim))) == 0, axis=0)))
           or (np.any(np.all(np.sum(ref3, axis=tuple(range(2, ref3.ndim))) == 0, axis=0))
           or np.any(np.all(np.sum(est3, axis=tuple(range(2, est3.ndim))) == 0, axis=0))))):
            
            os.remove("References/file0.wav")
            os.remove("Estimates/file0.wav")
            os.remove("References/file2.wav")
            os.remove("Estimates/file2.wav")
            
            # Calculate scores, save in "test.json"
            scores = museval.eval_dir("References", "Estimates", "Outputs")
            scores.save("test.json")

            # compute the metrics
            f = open('test.json',)

            # returns JSON object as
            # a dictionary
            data = json.load(f)

            # Iterating through the json
            # list
            # Metrics: Alarm
            jsonData = data['targets'][1]['frames'][0]['metrics']
            sdrTemp = jsonData.get('SDR')
            sirTemp = jsonData.get('SIR')
            sarTemp = jsonData.get('SAR')
            if (not (math.isnan(sdrTemp))):
                sdrA = sdrA + sdrTemp
                aCSDR = aCSDR + 1
            if (not (math.isnan(sirTemp))):
                sirA = sirA + sirTemp
                aCSIR = aCSIR + 1
            if (not (math.isnan(sarTemp))):
                sarA = sarA + sarTemp
                aCSAR = aCSAR + 1

            # Metrics: Dog
            jsonData = data['targets'][0]['frames'][0]['metrics']
            sdrTemp = jsonData.get('SDR')
            sirTemp = jsonData.get('SIR')
            sarTemp = jsonData.get('SAR')
            if (not (math.isnan(sdrTemp))):
                sdrD = sdrD + sdrTemp
                dCSDR = dCSDR + 1
            if (not (math.isnan(sirTemp))):
                sirD = sirD + sirTemp
                dCSIR = dCSIR + 1
            if (not (math.isnan(sarTemp))):
                sarD = sarD + sarTemp
                dCSAR = dCSAR + 1
            
        # Water dog
        # Error checking
        if (not ((np.any(np.all(np.sum(ref2, axis=tuple(range(2, ref2.ndim))) == 0, axis=0))
           or np.any(np.all(np.sum(est2, axis=tuple(range(2, est2.ndim))) == 0, axis=0)))
           or (np.any(np.all(np.sum(ref3, axis=tuple(range(2, ref3.ndim))) == 0, axis=0))
           or np.any(np.all(np.sum(est3, axis=tuple(range(2, est3.ndim))) == 0, axis=0))))):
            
            os.remove("References/file0.wav")
            os.remove("Estimates/file0.wav")
            os.remove("References/file1.wav")
            os.remove("Estimates/file1.wav")
            
            # Calculate scores, save in "test.json"
            scores = museval.eval_dir("References", "Estimates", "Outputs")
            scores.save("test.json")

            # compute the metrics
            f = open('test.json',)

            # returns JSON object as
            # a dictionary
            data = json.load(f)

            # Iterating through the json
            # list
            # Metrics: Water
            jsonData = data['targets'][1]['frames'][0]['metrics']
            sdrTemp = jsonData.get('SDR')
            sirTemp = jsonData.get('SIR')
            sarTemp = jsonData.get('SAR')
            if (not (math.isnan(sdrTemp))):
                sdrW = sdrW + sdrTemp
                wCSDR = wCSDR + 1
            if (not (math.isnan(sirTemp))):
                sirW = sirW + sirTemp
                wCSIR = wCSIR + 1
            if (not (math.isnan(sarTemp))):
                sarW = sarW + sarTemp
                wCSAR = wCSAR + 1

            # Metrics: Dog
            jsonData = data['targets'][0]['frames'][0]['metrics']
            sdrTemp = jsonData.get('SDR')
            sirTemp = jsonData.get('SIR')
            sarTemp = jsonData.get('SAR')
            if (not (math.isnan(sdrTemp))):
                sdrD = sdrD + sdrTemp
                dCSDR = dCSDR + 1
            if (not (math.isnan(sirTemp))):
                sirD = sirD + sirTemp
                dCSIR = dCSIR + 1
            if (not (math.isnan(sarTemp))):
                sarD = sarD + sarTemp
                dCSAR = dCSAR + 1
    
    # Delete files in "References" and "Estimates" folder for next interation
    files = glob.glob('Estimates/*.wav', recursive=True)
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))
    files = glob.glob('References/*.wav', recursive=True)
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

# Print out results
vcSDRAvg = sdrV / vcCSDR
vcSIRAvg = sirV / vcCSIR
vcSARAvg = sarV / vcCSAR
aSDRAvg = sdrA / aCSDR
aSIRAvg = sirA / aCSIR
aSARAvg = sarA / aCSAR
wSDRAvg = sdrW / wCSDR
wSIRAvg = sirW / wCSIR
wSARAvg = sarW / wCSAR
dSDRAvg = sdrD / dCSDR
dSIRAvg = sirD / dCSIR
dSARAvg = sarD / dCSAR
print("vcSDRAvg: ", vcSDRAvg)
print("vcSIRAvg: ", vcSIRAvg)
print("vcSARAvg: ", vcSARAvg)
print("aSDRAvg: ", aSDRAvg)
print("aSIRAvg: ", aSIRAvg)
print("aSARAvg: ", aSARAvg)
print("wSDRAvg: ", wSDRAvg)
print("wSIRAvg: ", wSIRAvg)
print("wSARAvg: ", wSARAvg)
print("dSDRAvg: ", dSDRAvg)
print("dSIRAvg: ", dSIRAvg)
print("dSARAvg: ", dSARAvg)

