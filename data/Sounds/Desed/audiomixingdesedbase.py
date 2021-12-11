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

cd ../../../../

# Grab files ending in ".wav" in "Alarm_bell_ringingCut"
alarmFilePaths = glob.glob("Alarm_bell_ringingCut/*.wav")
mixtureNumber = 0
for filepath in alarmFilePaths:
    # Get filename
    filename = filepath[14:(len(filepath) - 4)]
    # Get .wav file
    alarm = AudioSegment.from_file(filepath, format="wav")
    
    # Mix with a random file from VacuumCleanerCut
    vacuumFilePaths = glob.glob("Vacuum_cleanerCut/*.wav")
    index = int(random.random() * len(vacuumFilePaths))
    vacuumCleaner = AudioSegment.from_file(vacuumFilePaths[index], format="wav")
    
    # Overlay two sounds
    # Randomly pick first sound
    soundNum = int(random.random() * 2)
    if (soundNum == 0):
        combined = alarm.overlay(vacuumCleaner)
        alarmLabel = alarm
        vacuumCleanerLabel = vacuumCleaner
    else:
        combined = vacuumCleaner.overlay(alarm)
        vacuumCleanerLabel = vacuumCleaner
        alarmLabel = alarm
    
    name = str(mixtureNumber)+"1.wav"
    combined.export("MixturesBase/"+name , format='wav')
    vacuumCleanerLabel.export("MixturesBase/"+name+"Label/"+"2.wav" , format='wav')
    alarmLabel.export("MixturesBase/"+name+"Label/"+"3.wav" , format='wav')
    
    mixtureNumber = mixtureNumber + 1

