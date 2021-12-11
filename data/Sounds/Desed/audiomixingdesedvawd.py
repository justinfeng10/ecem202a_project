import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import tensorflow_io as tfio
# Pydub: https://github.com/jiaaro/pydub/blob/master/API.markdown
from pydub import AudioSegment
import csv
import glob
import random

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# For a given filepath ending with a wav file, obtain the waveform and the label
def get_waveform(label, filename, starttime, endtime):
    # Get .wav file
    waveform = AudioSegment.from_file(filename, format="wav")
    # Set Sample Width to 16
    waveform = waveform.set_sample_width(2)
    # Set number of channels to 1
    waveform = waveform.set_channels(1)
    # Get first 5 seconds
    if (len(waveform) >= 5000):
        waveform = waveform[:5000]
        # Cut clip into five 1-second chunks
        clip1 = waveform[:1000]
        clip2 = waveform[1000:2000]
        clip3 = waveform[2000:3000]
        clip4 = waveform[3000:4000]
        clip5 = waveform[4000:5000]
        # Save cut clips into new folder
        clip1.export(label+"Cut/1_"+filename, format="wav")
        clip2.export(label+"Cut/2_"+filename, format="wav")
        clip3.export(label+"Cut/3_"+filename, format="wav")
        clip4.export(label+"Cut/4_"+filename, format="wav")
        clip5.export(label+"Cut/5_"+filename, format="wav")

cd dataset/metadata/eval

tsv_file = open("public.tsv")
read_tsv = csv.reader(tsv_file, delimiter="\t")

def tsv_audio_data(label):
    name = list()
    begin = list()
    end = list()
    nameC = ""
    beginC = ""
    endC = ""
    prevName = ""
    nextName = ""
    flag = 0
    for row in read_tsv:
        if (row[3] == label and prevName != row[0] and flag == 0):
            nameC = row[0]
            beginC = row[1]
            endC = row[2]
            flag = 1
        elif (flag == 1 and row[0] != nameC):
            name.append(nameC)
            begin.append(beginC)
            end.append(endC)
            flag = 0
        prevName = row[0]
    return name, begin, end

cd ../../audio/eval/public

mkdir "Alarm_bell_ringingCut"

label = "Alarm_bell_ringing"
name, begin, end = tsv_audio_data(label)
print(len(name))
for i in range(0, len(name)):
    filename = name[i]
    starttime = begin[i]
    endtime = end[i]
    waveform = get_waveform(label, filename, starttime, endtime)

cd ../../../metadata/eval

tsv_file = open("public.tsv")
read_tsv = csv.reader(tsv_file, delimiter="\t")

cd ../../audio/eval/public

mkdir "Vacuum_cleanerCut"

label = "Vacuum_cleaner"
name, begin, end = tsv_audio_data(label)
print(len(name))
for i in range(0, len(name)):
    filename = name[i]
    starttime = begin[i]
    endtime = end[i]
    waveform = get_waveform(label, filename, starttime, endtime)

cd ../../../metadata/eval

tsv_file = open("public.tsv")
read_tsv = csv.reader(tsv_file, delimiter="\t")

cd ../../audio/eval/public

mkdir "Running_waterCut"

label = "Running_water"
name, begin, end = tsv_audio_data(label)
print(len(name))
for i in range(0, len(name)):
    filename = name[i]
    starttime = begin[i]
    endtime = end[i]
    waveform = get_waveform(label, filename, starttime, endtime)

cd ../../../metadata/eval

tsv_file = open("public.tsv")
read_tsv = csv.reader(tsv_file, delimiter="\t")

cd ../../audio/eval/public

mkdir "DogCut"

label = "Dog"
name, begin, end = tsv_audio_data(label)
print(len(name))
for i in range(0, len(name)):
    filename = name[i]
    starttime = begin[i]
    endtime = end[i]
    waveform = get_waveform(label, filename, starttime, endtime)

cd ../../../../

# Grab files ending in ".wav"
vacuumFilePaths = glob.glob("Vacuum_cleanerCut/*.wav")
alarmFilePaths = glob.glob("Alarm_bell_ringingCut/*.wav")
pouringFilePaths = glob.glob("Running_waterCut/*.wav")
dogFilePaths = glob.glob("DogCut/*.wav")
mixtureNumber = 0
# 1200 mixtures
for i in range(0, 1200):
    equal = True
    soundNum1 = 0
    soundNum2 = 0
    # Pick class
    while (equal):
        soundNum1 = int(random.random() * 4)
        soundNum2 = int(random.random() * 4)
        if (soundNum1 != soundNum2):
            equal = False
    
    filepath1 = ""
    filepath2 = ""
    # Randomly choose a file within a class
    if (soundNum1 == 0):
        filepath1 = random.choice(vacuumFilePaths)
    elif (soundNum1 == 1):
        filepath1 = random.choice(alarmFilePaths)
    elif (soundNum1 == 2):
        filepath1 = random.choice(pouringFilePaths)
    elif (soundNum1 == 3):
        filepath1 = random.choice(dogFilePaths)
    
    if (soundNum2 == 0):
        filepath2 = random.choice(vacuumFilePaths)
    elif (soundNum2 == 1):
        filepath2 = random.choice(alarmFilePaths)
    elif (soundNum2 == 2):
        filepath2 = random.choice(pouringFilePaths)
    elif (soundNum2 == 3):
        filepath2 = random.choice(dogFilePaths)
    
    # Get .wav file
    audio1 = AudioSegment.from_file(filepath1, format="wav")
    audio2 = AudioSegment.from_file(filepath2, format="wav")

    # Create mixture
    combined = audio1.overlay(audio2)
    label1 = audio1
    label2 = audio2
    
    # Save mixture and isolated sources
    name = str(mixtureNumber)+"1.wav"
    combined.export("MixturesVAWD/"+name , format='wav')
    audio1.export("MixturesVAWD/"+name+"Label/"+str(soundNum1)+".wav" , format='wav')
    audio2.export("MixturesVAWD/"+name+"Label/"+str(soundNum2)+".wav" , format='wav')
    
    # Isolated sources: empties
    if (soundNum1 != 0 and soundNum2 != 0):
        empty = tf.zeros([44100, 1])
        emptyWav = tf.audio.encode_wav(empty, 44100, name=None)
        tf.io.write_file("MixturesVAWD/"+name+"Label/"+str(0)+".wav", emptyWav, name=None)
    if (soundNum1 != 1 and soundNum2 != 1):
        empty = tf.zeros([44100, 1])
        emptyWav = tf.audio.encode_wav(empty, 44100, name=None)
        tf.io.write_file("MixturesVAWD/"+name+"Label/"+str(1)+".wav", emptyWav, name=None)
    if (soundNum1 != 2 and soundNum2 != 2):
        empty = tf.zeros([44100, 1])
        emptyWav = tf.audio.encode_wav(empty, 44100, name=None)
        tf.io.write_file("MixturesVAWD/"+name+"Label/"+str(2)+".wav", emptyWav, name=None)
    if (soundNum1 != 3 and soundNum2 != 3):
        empty = tf.zeros([44100, 1])
        emptyWav = tf.audio.encode_wav(empty, 44100, name=None)
        tf.io.write_file("MixturesVAWD/"+name+"Label/"+str(3)+".wav", emptyWav, name=None)
    mixtureNumber = mixtureNumber + 1

