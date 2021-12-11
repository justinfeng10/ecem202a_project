# Audio Source Separation on Arduino

Welcome to the Github repository for the Audio Source Separation on Arduino Project, for ECE M202A.

Here is the link to the project website: https://justinfeng10.github.io/ecem202a_project/

* doc/ contains website content
* software/ contains the code used in this project
* data/ contains the dataset

## Examples:
To run test samples on the trained ML models (this example is for the VAWD model, "Model 2" from the report), run this code, assuming one is in the directory that contains "SourceSeparationTest.py" and there is a subfolder called "SamplesVAWD" that has the input mixture named "VacuumAlarm.wav":

python3 SourceSeparationTest.py SamplesVAWD/VacuumAlarm.wav

This script grabs the input sound file from the folder "SamplesVAWD", runs inference, and prints the input spectrogram and the predicted output spectrograms. Additionally, the corresponding wav files will be placed in the respective model folder under the created folder "SourceSeparationTestOutput". One can run examples for the other models (base and VAWDComplex) by using the different python codes as shown below in the examples. One can also run their own input mixtures by placing  a one second wav file within the respective subfolder for the model one desires to test.

Examples:
Model 1: python3 SourceSeparationTestBase.py SamplesBase/VacuumAlarm1.wav
Model 3: python3 SourceSeparationTestComplex.py SamplesVAWDComplex/VacuumAlarm.wav

## Abstract:
When walking down a street, one may hear different musical instruments being played at the same time. One may wonder how to “unmix” the musical sounds into their component parts. Audio source separation is the process of separating a mixture (e.g. a pop band recording) into isolated sounds from individual sources (e.g. just the lead vocals). In this project, an environmental audio source separation model using a convolutional neural network was developed that can separate an audio mixture of up to two different audio sources at the same time (e.g. vacuum and alarm). Live audio can then be recorded through the Arduino Nano 33 BLE Sense’s on-board microphone and passed to the machine learning model for inference.
