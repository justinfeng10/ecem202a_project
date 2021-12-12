# Audio Source Separation on Arduino

Welcome to the Github repository for the Audio Source Separation on Arduino Project, for ECE M202A.

Here is the link to the project website: https://justinfeng10.github.io/ecem202a_project/

* doc/ contains website content
* software/ contains the code used in this project
* data/ contains the dataset

## Examples

To run test samples on the trained ML models (this example is for the VAWD model, "Model 2" from the report), run this code, assuming one is in the directory that contains "SourceSeparationTest.py" and there is a subfolder called "SamplesVAWD" that has the input mixture named "VacuumAlarm.wav":

python3 SourceSeparationTest.py SamplesVAWD/VacuumAlarm.wav

This script grabs the input sound file from the folder "SamplesVAWD", runs inference, and prints the input spectrogram and the predicted output spectrograms. Additionally, the corresponding wav files will be placed in the respective model folder under the created folder "SourceSeparationTestOutput". One can run examples for the other models (base and VAWDComplex) by using the different python codes as shown below in the examples. One can also run their own input mixtures by placing a one second wav file (with sampling rate of either 16 kHz or 44.1 kHz) within the respective subfolder for the model one desires to test.

Examples:

Model 1: python3 SourceSeparationTestBase.py SamplesBase/VacuumAlarm1.wav

Model 2: python3 SourceSeparationTest.py SamplesVAWD/VacuumAlarm.wav

Model 3: python3 SourceSeparationTestComplex.py SamplesVAWDComplex/VacuumAlarm.wav

## Required Submissions

* [Proposal](/docs/proposal.md)
* [Midterm Checkpoint Presentation Slides](https://docs.google.com/presentation/d/1Vsg-iq3j5DP994vDR3WmkYU4yzW6A_noixU22QlIx8o/edit?usp=sharing)
* [Final Presentation Slides](https://docs.google.com/presentation/d/1Yf8Y32Tk36Zz1VE6MUDrayRQ5ihTwDeXtkDPviiufFo/edit?usp=sharing)
* [Final Report](/docs/report.md)
* [Final Report Video](https://youtu.be/9TLBRLbjA2U)
* [Demo 1](https://youtu.be/1CPYJBp5IpI)
* [Demo 2](https://youtu.be/DU1GsaJTA9M)

## Abstract

When walking down a street, one may hear different musical instruments being played at the same time. One may wonder how to “unmix” the musical sounds into their component parts. Audio source separation is the process of separating a mixture (e.g. a pop band recording) into isolated sounds from individual sources (e.g. just the lead vocals). In this project, an environmental audio source separation model using a convolutional neural network was developed that can separate an audio mixture of up to two different audio sources at the same time (e.g. vacuum and alarm). Live audio can then be recorded through the Arduino Nano 33 BLE Sense’s on-board microphone and passed to the machine learning model for inference.

## References

[1] P. Chandna, M. Miron, J. Janer, and E. Gomez. Monoaural Audio Source Separation Using Deep Convolutional Neural Networks. LVA/ICA, Feb. 2017. doi:
    10.1007/978-3-319-53547-0_25. Available:
    <https://www.researchgate.net/publication/313732034_Monoaural_Audio_Source_Separation_Using_Deep_Convolutional_Neural_Networks>
    
[2] E. Vincent, R. Gribonval, and C. Fevotte. Performance measurement in blind audio source separation. IEEE Transactions on Audio, Speech, and Language
    Processing, 2006. doi: 10.1109/TSA.2005.858005. Available: <https://ieeexplore.ieee.org/document/1643671>
    
[3] Q. Kong, Y. Cao, H. Liu, K. Choi, and Y. Wang. Decoupling Magnitude and Phase Estimation with Deep ResUNet for Music Source Separation. ISMIR, 2021.
    arXiv:2109.05418. Available: <https://arxiv.org/abs/2109.05418>
    
[4] K. M. Jeon, C. Chun, G. Kim, C. Leem, B. Kim and W. Choi. Lightweight U-Net Based Monaural Speech Source Separation for Edge Computing Device. ICCE, 2020.
    doi: 10.1109/ICCE46568.2020.9043051. Available: <https://ieeexplore.ieee.org/document/9043051>
    
[5] K. J. Piczak. ESC: Dataset for Environmental Sound Classification. Proceedings of the 23rd Annual ACM Conference on Multimedia, 2015. Available:
    <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YDEPUT>
    
[6] N. Turpault, R. Serizel, A. Parag Shah, and J. Salamon. Sound event detection in domestic environments with weakly labeled data and soundscape synthesis.
    Workshop on Detection and Classification of Acoustic Scenes and Events, Oct. 2019. Available: <https://project.inria.fr/desed/>
    
[7] jiaaro. Pydub. Available: <https://pydub.com/>

[8] museval. Available: <https://sigsep.github.io/sigsep-mus-eval/>

[9] S. Yao et al. STFNets: Learning Sensing Signals from the Time-Frequency Perspective with Short-Time Fourier Neural Networks. W3C2, 2019. arXiv:1902.07849. 
    Available: <https://arxiv.org/abs/1902.07849>

[10] Simple audio recognition: Recognizing keywords. TensorFlow. Available: <https://tensorflow.org/tutorials/audio/simple_audio>
    
[11] Arduino IDE. Available: <https://www.arduino.cc/en/software>

[12] Jupyter Notebook. Available: <https://jupyter.org/>

[13] Google Colab. Available: <https://colab.research.google.com/>
