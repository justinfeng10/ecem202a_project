# Pydub: https://github.com/jiaaro/pydub/blob/master/API.markdown
from pydub import AudioSegment
import glob
import random

mkdir "VacuumCleanerCut"

import glob
# Grab files ending in ".ogg" in "VacuumCleaner"
vacuumFilePaths = glob.glob("VacuumCleaner/*.ogg")
for filepath in vacuumFilePaths:
    # Get filename
    filename = filepath[14:(len(filepath) - 4)]
    # Get .ogg file
    vacuumCleaner = AudioSegment.from_file(filepath, format="ogg")
    # Set frame rate to 16 kHz if so choose
    #vacuumCleaner = vacuumCleaner.set_frame_rate(16000)
    # Set Sample Width to 16
    vacuumCleaner = vacuumCleaner.set_sample_width(2)
    # Cut clip into five 1-second chunks
    clip1 = vacuumCleaner[:1000]
    clip2 = vacuumCleaner[1000:2000]
    clip3 = vacuumCleaner[2000:3000]
    clip4 = vacuumCleaner[3000:4000]
    clip5 = vacuumCleaner[4000:5000]
    # Save cut clips into new folder
    clip1.export("VacuumCleanerCut/1_"+filename+".wav", format="wav")
    clip2.export("VacuumCleanerCut/2_"+filename+".wav", format="wav")
    clip3.export("VacuumCleanerCut/3_"+filename+".wav", format="wav")
    clip4.export("VacuumCleanerCut/4_"+filename+".wav", format="wav")
    clip5.export("VacuumCleanerCut/5_"+filename+".wav", format="wav")

mkdir "AlarmCut"

# Grab files ending in ".ogg" in "Alarm"
pouringFilePaths = glob.glob("Alarm/*.ogg")
for filepath in pouringFilePaths:
    # Get filename
    filename = filepath[6:(len(filepath) - 4)]
    # Get .ogg file
    pouringWater = AudioSegment.from_file(filepath, format="ogg")
    # Set Sample Width to 16
    pouringWater = pouringWater.set_sample_width(2)
    # Cut clip into five 1-second chunks
    clip1 = pouringWater[:1000]
    clip2 = pouringWater[1000:2000]
    clip3 = pouringWater[2000:3000]
    clip4 = pouringWater[3000:4000]
    clip5 = pouringWater[4000:5000]
    # Save cut clips into new folder
    clip1.export("AlarmCut/1_"+filename+".wav", format="wav")
    clip2.export("AlarmCut/2_"+filename+".wav", format="wav")
    clip3.export("AlarmCut/3_"+filename+".wav", format="wav")
    clip4.export("AlarmCut/4_"+filename+".wav", format="wav")
    clip5.export("AlarmCut/5_"+filename+".wav", format="wav")

mkdir "PouringWaterCut"

# Grab files ending in ".ogg" in "PouringWater"
pouringFilePaths = glob.glob("PouringWater/*.ogg")
for filepath in pouringFilePaths:
    # Get filename
    filename = filepath[13:(len(filepath) - 4)]
    # Get .ogg file
    pouringWater = AudioSegment.from_file(filepath, format="ogg")
    # Set Sample Width to 16
    pouringWater = pouringWater.set_sample_width(2)
    # Cut clip into five 1-second chunks
    clip1 = pouringWater[:1000]
    clip2 = pouringWater[1000:2000]
    clip3 = pouringWater[2000:3000]
    clip4 = pouringWater[3000:4000]
    clip5 = pouringWater[4000:5000]
    # Save cut clips into new folder
    clip1.export("PouringWaterCut/1_"+filename+".wav", format="wav")
    clip2.export("PouringWaterCut/2_"+filename+".wav", format="wav")
    clip3.export("PouringWaterCut/3_"+filename+".wav", format="wav")
    clip4.export("PouringWaterCut/4_"+filename+".wav", format="wav")
    clip5.export("PouringWaterCut/5_"+filename+".wav", format="wav")

mkdir "DogCut"

# Grab files ending in ".ogg" in "Dog"
pouringFilePaths = glob.glob("Dog/*.ogg")
for filepath in pouringFilePaths:
    # Get filename
    filename = filepath[4:(len(filepath) - 4)]
    # Get .ogg file
    pouringWater = AudioSegment.from_file(filepath, format="ogg")
    # Set Sample Width to 16
    pouringWater = pouringWater.set_sample_width(2)
    # Cut clip into five 1-second chunks
    clip1 = pouringWater[:1000]
    clip2 = pouringWater[1000:2000]
    clip3 = pouringWater[2000:3000]
    clip4 = pouringWater[3000:4000]
    clip5 = pouringWater[4000:5000]
    # Save cut clips into new folder
    clip1.export("DogCut/1_"+filename+".wav", format="wav")
    clip2.export("DogCut/2_"+filename+".wav", format="wav")
    clip3.export("DogCut/3_"+filename+".wav", format="wav")
    clip4.export("DogCut/4_"+filename+".wav", format="wav")
    clip5.export("DogCut/5_"+filename+".wav", format="wav")

# Grab files ending in ".ogg"
vacuumFilePaths = glob.glob("VacuumCleanerCut/*.wav")
alarmFilePaths = glob.glob("AlarmCut/*.wav")
pouringFilePaths = glob.glob("PouringWaterCut/*.wav")
dogFilePaths = glob.glob("DogCut/*.wav")
mixtureNumber = 0
# 800 mixtures
for i in range(0, 800):
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
    
    # Get .ogg file
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

