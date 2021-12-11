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

# Grab files ending in ".ogg" in "PouringWater"
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

# Grab files ending in ".ogg" in "PouringWaterCut"
pouringFilePaths = glob.glob("AlarmCut/*.wav")
mixtureNumber = 0
for filepath in pouringFilePaths:
    # Get filename
    filename = filepath[14:(len(filepath) - 4)]
    # Get .ogg file
    pouringWater = AudioSegment.from_file(filepath, format="wav")
    
    # Mix with a random file from VacuumCleanerCut
    vacuumFilePaths = glob.glob("VacuumCleanerCut/*.wav")
    for i in range(1, 4):
        index = int(random.random() * len(vacuumFilePaths))
        vacuumCleaner = AudioSegment.from_file(vacuumFilePaths[index], format="wav")

        combined = pouringWater.overlay(vacuumCleaner)
        pouringWaterLabel = pouringWater
        vacuumCleanerLabel = vacuumCleaner

        name = str(mixtureNumber)+ str(i) + "1.wav"
        combined.export("MixturesBase/"+name , format='wav')
        vacuumCleanerLabel.export("MixturesBase/"+name+"Label/"+"2.wav" , format='wav')
        pouringWaterLabel.export("MixturesBase/"+name+"Label/"+"3.wav" , format='wav')

    mixtureNumber = mixtureNumber + 1

