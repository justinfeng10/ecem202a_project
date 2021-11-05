# Project Proposal

## 1. Motivation & Objective

In this project, the goal is to separate two target audio signals (e.g. faucet and microwave) using one machine learning model. In audio classification, detecting the presence of multiple target sounds may be desired. Additionally, each target sound may be present with other target sounds. The aim is to separate this "mixture" of sounds into their component target sounds in one model, recognizing when each are present (or not present). Existing work on source separation on embedded systems is limited, creating motivation to successfully implement source separation on an Arduino.

## 2. State of the Art & Its Limitations

The Audio source separation pipeline consists of a few steps. Given a mixture, create a spectrogram (via a short-time Fourier transform) and pass it through a convolutional neural network (CNN). The CNN's output is then passed through a time frequency masking step and an inverse short-time Fourier transform in order to obtain the separated sources [1].

One limitation is the lack of support for edge devices. A large portion of existing work does not consider resource constraints of edge devices such as limited memory and processing power. Some existing work that considers resource constraints does not separate two distinct target signals (excluding noise) [2] or lacks real implementation on an edge device [3].

## 3. Novelty & Rationale

The novelty in my approach is to consider the Arduino's constraints from the beginning in the creation of an audio source separation model and to provide tangible deployment and test results. I believe my approach will be successful because previous work can be leveraged in addition to an understanding of the Arduino's capabilities to fine tune a model as well as implement audio processing steps that fit under the Arduino's constraints.

## 4. Potential Impact

A successful result will enable edge devices to classify audio samples more robustly. A difficulty in audio recognition is the ability to parse multiple target sounds at the same time. Successful work in this project can be expanded to separate more target sounds or handle even more complicated problems such as speech separation on edge devices.

## 5. Challenges

A main challenge is handling the resource constrained nature of the Arduino board. The Arduino board has limited memory, constraining both our machine learning model and our input data. The processing rate is limited and the impact of delays will need to be considered. Additionally, the Arduino board's microphone can only sample at 16kHz. All these constraints may lead to difficulties deploying the TensorFlow model onto the Arduino. Moreover, source separation itself is a difficult problem, so obtaining results in TensorFlow could be a bottleneck. Both a quality dataset and a fine-tuned machine learning model are needed. Determining the efficacy of the model is difficult as well. Tactics discussed in the "Metrics of Success" paragraph will be needed.

## 6. Requirements for Success

A thorough understanding of machine learning models and audio data preprocessing is necessary for this project. Audio signals need to be processed in a manner that the machine learning model can understand, and quality audio signals are needed to begin with. The machine learning model then needs to properly learn this separation problem and fine tune accordingly. Additionally, a thorough understanding of the Arduino board itself (memory requirements, processing ability, microphone, TinyML, etc.) will allow for a successful deployment of the model.

## 7. Metrics of Success

Since accuracy is not a metric of success in source separation, additional metrics will be needed. Objective metrics such as the source-to-distortion ratio can be used to analyze the quality of the sources themselves. On both the Arduino and the TensorFlow model, tests could be run to see if the model can correctly predict the present target sounds under various circumstances (one target present, both targets present, transitions between the two targets, etc.). Subjective metrics such as analyzing the resulting spectrograms or playing back the separate signals could be used as well. 

## 8. Execution Plan

The first step is to successfully process audio data for the machine learning model. Processing includes creating uniformity in the number of channels, sampling rate, and time length of various audio files as well as creating spectrograms. The next step is to create a machine learning model in TensorFlow that can successfully separate two signals. Once a successful model is obtained, the following step is to process audio data on the Arduino via the board's microphone (using similar processing steps as discussed before) and pass it through the TensorFlow model to see if the results are comparable. Once this is verified, the TensorFlow model can then be deployed on the Arduino board via TensorFlow Lite to run live tests on the Arduino itself.

## 9. Related Work

### 9.a. Papers

**1) Monoaural Audio Source Separation Using Deep Convolutional Neural Networks [1]**\
This paper details a neural network approach to source separation. The paper discusses audio preprocessing techniques as well as a network structure that leads to successful source separation. The model detailed here can be used as a template model for my source separation implementation.

**2) Lightweight U-Net Based Monaural Speech Source Separation for Edge Computing Device [2]**\
This paper discusses a lightweight network using U-Net, a network typically used in medical image separation. The writers apply this network to source separation, separating speech from noise on an edge computing device. I can leverage this work to enable the separation of two tangible target sounds on Arduino.

**3) Sudo Rm-Rf Efficient Networks for Universal Audio Source Separation [3]**\
This paper highlights an approach to reduce the resource usage of a source separation model. The network provided specifically attempts to reduce memory requirements. This paper does not provide a tangible implementation on an edge device, so I can use strategies discussed in this paper to create a model that fits onto the Arduino board.

### 9.b. Datasets

I plan to use Freesound [4] and BBC Sound Effects [5]. Freesound and BBC Sound Effects contain a diverse set of sounds that can be used legally by either citing or attributing to the website and/or uploader. Data augmentation can also be used to expand the dataset if needed.

### 9.c. Software

I will use the Arduino IDE [6] to write code for the Arduino board. I will use Jupyter Notebook [7] and/or Google Colab [8]to create and debug my TensorFlow model. I also plan to use the TensorFlow Lite converter (Python API) [9] to convert my TensorFlow model into TensorFlow Lite. Lastly, software such as Edge Impulse [10] can be used to assist in any of these steps.

## 10. References

[1] P. Chandna, M. Miron, J. Janer, and E. Gomez, "Monoaural Audio Source Separation Using Deep Convolutional Neural Networks," LVA/ICA 2017. Feb. 2017, doi: 10.1007/978-3-319-53547-0_25. Available: https://www.researchgate.net/publication/313732034_Monoaural_Audio_Source_Separation_Using_Deep_Convolutional_Neural_Networks

[2] K. M. Jeon, C. Chun, G. Kim, C. Leem, B. Kim and W. Choi, "Lightweight U-Net Based Monaural Speech Source Separation for Edge Computing Device," 2020 IEEE ICCE, 2020, pp. 1-4, doi: 10.1109/ICCE46568.2020.9043051. Available: https://ieeexplore.ieee.org/document/9043051

[3] E. Tzinis, Z. Wang, and P. Smaragdis, "Sudo Rm-Rf: Efficient Networks for Universal AUdio Source Separation," 2020 IEEE 30th International Workshop on MLSP, 2020, doi: 10.1109/MLSP49062.2020.9231900. Available: https://arxiv.org/abs/2007.06833

[4] Freesound. Available: https://freesound.org/

[5] BBC Sound Effects. Avaliable: https://sound-effects.bbcrewind.co.uk/

[6] Arduino IDE. Available: https://www.arduino.cc/en/software

[7] Jupyter Notebook. Available: https://jupyter.org/

[8] Google Colab. Available: https://colab.research.google.com/

[9] TensorFlow Lite Converter. Available: https://www.tensorflow.org/lite/convert/

[10] Edge Impulse. Available: https://www.edgeimpulse.com/
