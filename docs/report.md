# Table of Contents
* Abstract
* [Introduction](#1-introduction)
* [Related Work](#2-related-work)
* [Technical Approach](#3-technical-approach)
* [Evaluation and Results](#4-evaluation-and-results)
* [Discussion and Conclusions](#5-discussion-and-conclusions)
* [References](#6-references)
* [Contributions](#7-contributions)

# Abstract

When walking down a street, one may hear different musical instruments being played at the same time. One may wonder how to “unmix” the musical sounds into their component parts. Audio source separation is the process of separating a mixture (e.g. a pop band recording) into isolated sounds from individual sources (e.g. just the lead vocals). In this project, an environmental audio source separation model using a convolutional neural network was developed that can separate an audio mixture of up to two different audio sources at the same time (e.g. vacuum and alarm). Live audio can then be recorded through the Arduino Nano 33 BLE Sense’s on-board microphone and passed to the machine learning model for inference.

# 1. Introduction

In this project, the goal is to separate a mixture of two target audio signals (e.g. vacuum and alarm) using one machine learning model. In audio classification, detecting the presence of multiple target sounds may be desired, with different target sounds being present at the same time. The aim is to separate this "mixture" of sounds into its component target sounds in one model.

A source separation model generally follows a specific structure: given a mixture, create a spectrogram (via a short-time Fourier transform) and pass it through a convolutional neural network (CNN), which first encodes the data and then decodes the data and makes predictions on the separated sources. One limitation is the lack of support for edge devices. A large portion of existing work does not consider resource constraints of edge devices such as limited memory and processing power. Existing work also does not consider isolated single sources as input to the model (instead of just mixtures), as well as any implemented connection to a resource constrained device.

In this project, an existing model for music source separation was utilized and adapted to environmental audio source separation. Our model is able to handle mixtures, as well as individual audio sources. Additionally, further adjustments are made such as complex value capabilities and quantization in order to expand upon the original model. The model is also developed with the Arduino Nano 33 BLE Sense in mind. For example, data preprocessing is designed such that the model is optimized for a resource constrained device (limited sample rate and size of data). Live audio can be recorded directly on the Arduino and passed to the machine learning model on a computer for inference.

A successful result will provide an ability for more complex source separation models to be developed and a foundation for edge devices to classify audio samples more robustly. Another impact would be the improvement of processing capacity for more complicated work such as real-time speech recognition and separation.

Challenges exist in both the source separation model implementation and the Arduino. For the source separation model, it is challenging to understand the machine learning models themselves and the intricacies behind these models such as the loss function and processing of the output. It is also challenging to create a dataset large enough to provide robustness for our model. For the Arduino, the main challenge is to collect the entirety of one second of audio samples (16000) and send them to the model due to the nature of the PDM module which by default only receives 256 samples each time. It is also important to convert PDM to PCM before passing it to the model because the model can only read PCM.

To ensure the success of our project, a few considerations need to be made. For the source separation model, a robust dataset, proper processing, complete machine learning model, and proper metrics are needed to both create and evaluate the model. For the Arduino, it is important to make sure a second of samples are completely collected and there is no corrupted data during the collection process.

The quality of this source separation project can be determined both qualitatively and quantitatively. Qualitative analysis includes listening to the model output in the time domain as well as viewing the output spectrograms. Quantitative analysis includes calculating source separation metrics commonly used in the source separation community. The quality of the Arduino can be determined by whether the output of audio sampled through the on-board microphone is clear and reaches one second. 

# 2. Related Work

In the paper “Monoaural Audio Source Separation Using Deep Convolutional Neural Networks” [1], the authors design an efficient model and pipeline for audio source separation. They utilize a CNN that consists of an encoding and a decoding stage to separate the sound sources. Given a mixture, they compute the short time fourier transform (STFT) and pass the magnitude of the resulting spectrogram into the CNN. After applying a time frequency mask, they obtain the separated sources by computing the inverse short time fourier transform (ISTFT).

The next paper [2] provides seminal metrics for audio source separation. In order to qualitatively evaluate the success of a source separation algorithm, they develop four metrics: source-to-distortion ratio (SDR), source-to-interferences ratio (SIR), sources-to-noise ratio (SNR), and sources-to-artifacts ratio (SAR). These metrics return a value in decibels. The higher the value, the greater the separation achieved as the energy of the source is greater than the distortion for instance in SDR.

This paper [3] discusses approaches to handle magnitude and phase of input spectrograms to source separation models. They discuss a few approaches, such as a magnitude mask and the use of the mixture phase to recover the separated sources’ ISTFT, which paper [1] applies, as well as approaches that consider complex estimation. 

This paper [4] discusses the separation of speech from noise. They utilize a U-Net style architecture, similar to the other papers. The paper discusses a Wiener filter similar to the time frequency mask used in paper [1] as well as a similar loss function. The authors describe the source separation process and the type of data needed for the machine learning model, including the mixture as well as the original isolated sources. This paper also discusses approaches to integrate models onto edge devices, such as quantization to 8-bit values.

# 3. Technical Approach

## Dataset

The first dataset utilized in this project was ESC-50, under the collective dataset titled “ESC: Dataset for Environmental Sound Classification”, downloaded from the Harvard Dataverse and created by [5]. This dataset consists of 50 classes, with 40 clips per class. Clips in this dataset are clean and consistent throughout the clip.

The second dataset utilized in this project was DESED: Domestic Environment Sound Event Detection Dataset [6]. This dataset was part of the DCASE challenge (Detection and Classification of Acoustic Scenes and Events). Clips in this dataset contain more time and sound variation and also more interference, either from other sounds or noise itself. This dataset will provide robustness to the machine learning model.

To curate a dataset that the machine learning model can use, one second clips of sounds were isolated and downsampled to 16 kHz, keeping the Arduino in mind. We designed the model such that the Arduino could collect input and pass it to the model. The Arduino microphone has a 16 kHz sampling frequency and can manage smaller sample lengths considering the slow transmission rate over the serial port and limited memory. 

Next, mixtures were created utilizing the Pydub library [7]. This library allows one to overlay one sound over the other over one channel. The one second clips created in the first step contain time and frequency variations that when combined at random, create diversity within the mixture set. To create the mixture dataset, two sources were chosen at random and overlaid. The mixture dataset has time and frequency diversity, and the volume of each individual clip varies.

In the end, our dataset had a maximum of 2400 one second clips, and these resulting clips were modified from the preexisting clips downloaded from the sources. This data is then split into sets for the machine learning model.

## Spectrogram

Before the dataset can be passed to the ML model, the STFT needs to be computed. The STFT computes the fourier transform over successive windows, with a given hopsize. There is a tradeoff between time and frequency resolution that needs to be considered. As the fourier transform size increases, higher frequency resolution is obtained (greater number of frequency bins) at the expense of time resolution. As the fourier transform size decreases, higher time resolution is obtained at the expense of frequency resolution (lower number of frequency bins). In the STFT, window size is usually a power of two to increase computation speed. The hop size determines where in time is the next STFT taken. Experimentally, parameters of window size = 512 (3.2% of a clip) and hop size = 128 (0.8% of a clip) were chosen. By experimenting with different parameters, one can achieve an optimal tradeoff between time and frequency resolution. In Figure 1, from left to right, are spectrograms for a vacuum cleaner, alarm, water, and a dog. The STFT is able to capture unique characteristics for each sound.

![Figure 1](https://github.com/justinfeng10/ecem202a_project/blob/main/docs/media/Figure1.JPG)

## ML Model

The model designed in this project (Figure 2) is based on the approach discussed in paper [1]. Each piece of data consists of the magnitude of the input sound spectrogram, the spectrograms of the isolated sources, similar to [4] (used for the loss calculation), and the phase of the input sound (used to calculate the ISTFT). The model itself consists of an encoding stage and a decoding stage. First, the encoding stage computes vertical and horizontal convolutions across the input, creating a latent representation of the source. The vertical and horizontal convolutions learn different timbre information and temporal changes, respectively. Before the latent representation is sent to the decoding stage, it is sent to a dense layer. This bottleneck reduces the dimensionality of the input. Then, the latent representation is sent to the decoding stage, where the separated representations are formed. Transpose convolutions are then applied to the latent representation to prepare the output for the time frequency masks. Parameters for our model were experimentally determined through thorough testing.

Time frequency masks are then applied to construct the separated sources. The equation, shown in Figure 3,  consists of an output source divided by the sum of the other output sources. Once masks are calculated, the mask can then be multiplied by the model input spectrogram to receive the predicted spectrogram. 

At this point, the loss needs to be calculated. Loss is defined as the mean squared error between the predicted output source and the provided label source. These loss terms are reduced via sum and added together. Additionally, an extra loss term can be added that subtracts out the mean squared error between predicted sources, multiplied by a constant “alpha”. The loss function is shown in Figure 4. 

After separation, the ISTFT can be calculated to convert the output spectrograms to audio. The phase of the input sound is used as an estimation of the phase of the outputs. The equation in Figure 5 can be used to convert the magnitude spectrogram output of the estimated source to a form that the ISTFT can use, using the magnitude spectrogram output of the specific source and the input mixture phase to recoup the real and imaginary components. Once this is completed, one can calculate the ISTFT.

Two case studies were further investigated. The first case study involves complex numbers. Similar to paper [3], the phase information is trained on as well as the magnitude. Instead of having only the magnitude as input and utilizing the input phase as an estimate of the output phase, both the magnitude and phase are passed into the model to be trained on. Approaches such as those discussed in STFNets [8] that adjust the convolution (such as padding) based on the properties of a spectrogram with real and imaginary components were unattainable in this project as the model implemented cannot take in imaginary numbers as input. Thus, our implemented approach using magnitude and phase is an attempt to bridge the gap between the approach discussed in [1] and the approach discussed in [8]. 

The second case study involves quantization, similar to paper [4]. 8-bit fixed point quantization was utilized at intermediate feature map results (word length of eight, fraction length of four) in order to analyze how a reduction in value resolution affects the output.

## Arduino

Using the on-board microphone of Arduino Nano 33 BLE Sense, it is possible to sample audio signals without an external microphone module. Figure 6 illustrates the process from collecting data to sending them to the ML model. 

By default, the PDM module in Arduino IDE only collects 512 bytes at a time, allowing one to have 256 two byte samples in the sample buffer. Therefore, it is necessary to increase the PDM buffer size to 32000 bytes (16000 samples times 2 bytes) in order to store one second worth of audio. Before the data is stored in the buffer, the data is converted to PCM format, a format that the ML model can understand. While waiting for the commands, the Arduino keeps rolling in new data until user input is received. Once user input is received, the arduino will send the samples to the computer through the “Serial” module. After that, it is ready to be imported into an array and written into a wav file. Now the model is able to analyze the imported file and can start the separation process. 

# 4. Evaluation and Results

Four models were simulated. Model 1 separates two sources, with two classes: vacuum and alarm. Model 2 expands on Model 1 to separate two sources, with four classes: vacuum, alarm, water, and dog. Model 3 modifies Model 2 by accounting for complex values, and Model 4 modifies Model 2 by quantizing the intermediate feature maps.

## Model 1: Vacuum and Alarm

In Figure 7, a mixture consisting of a vacuum cleaner and an alarm sound is shown (picture 1). We see the predicted vacuum output (picture 2) and the predicted alarm output (picture 3). Note that in picture 2, we see the general shading in the spectrogram and the notable horizontal line across time that signifies a vacuum cleaner. Note that in picture 3, we see the frequency harmonics and shorter bursts of alarm sound.

In Figure 8, the resulting SDR, SIR, and SAR are shown, in decibels. The higher the value, the better. The vacuum cleaner is being separated slightly better than the alarm. Both are being separated relatively well.

One can also do a qualitative analysis of the sounds. Test cases can be run using the “SourceSeparationTestBase.py” file.

## Model 2: Vacuum, Alarm, Water, Dog

In Figure 9, a mixture of a vacuum cleaner and an alarm (picture 1) is shown again to compare to Model 1. The model is able to detect that the mixture consists of a vacuum cleaner and an alarm, while not having any water or dog sounds. Within picture 2 and picture 3, definite separation can be seen, using a similar analysis as for Model 1. Figure 10 shows a mixture of a vacuum cleaner and a dog being separated.

In Figure 11, the resulting metrics are shown. Vacuum cleaner sounds still perform the best comparatively. Alarm experiences a slight drop in values. Dog performs relatively close to vacuum cleaner, while water performs closer to alarm. One can surmise that since vacuum cleaners and water share a more similar time-frequency profile than the other sounds, the model is able to distinguish one better than the other. 

One can also do a qualitative analysis of the sounds. Test cases can be run using the “SourceSeparationTest.py” file.

## Model 3: Complex

Compared to Models 1 and 2, the model has a bit more trouble separating the sources. Particularly, in Figure 13, we notice that the model is having some trouble between alarm (picture 3) and water (picture 4). In Figure 14, the metrics are also slightly down in all categories. This shows that estimating the phase is complex.

One can also do a qualitative analysis of the sounds. Test cases can be run using the “SourceSeparationTestComplex.py” file.

## Model 4: Quantized

By quantizing the intermediate feature maps to 8-bit fixed point, the metrics change. Interestingly, the model struggles to separate water, but is comparatively strong in the other classes.

Note that the model could not be saved unlike the other models, but in the “QuantizedExamples” Folder are sample inputs and outputs of the quantized model.

## Arduino

To evaluate if the recorded audio is valid, a wav file called “temp.wav” will be generated once the sampling process is finished. This wav file should be 1 second long in order to successfully be separated by the model. After a few rounds of testing, the results met the expectations. The program was able to generate spectrograms using the file and run the ML model. Figure 17 shows the spectrogram output of a live test of a Vacuum and Alarm mixture.

# 5. Discussion and Conclusions

Source separation is a unique challenge. How can a machine learning model learn what makes a sound unique? How can it distinguish between sounds that are similar to each other? In this project, we successfully achieved two source separation. Given an input mixture, the model outputs the predicted separated sounds. Many limitations and challenges were faced. First of all, source separation is a difficult problem and a lot of time was spent to get the base model up and running. This did not leave much time for further optimizations and improvements. A limitation of the model itself includes an increased difficulty in separating mixtures that contain similar aspects of different classes. For instance, aspects of a vacuum cleaner may share similar spectrogram characteristics of that of water. This is a fundamental limit of this spectrogram approach. Future work could utilize techniques found in [8] such as spectral padding to adjust the convolution as well as hologram interleaving in order to optimize the time-frequency trade off. Other work could consist of working on time domain data entirely.

# 6. References

[1] P. Chandna, M. Miron, J. Janer, and E. Gomez. Monoaural Audio Source Separation Using Deep Convolutional Neural Networks. LVA/ICA, Feb. 2017. doi:
    10.1007/978-3-319-53547-0_25. Available:
    <https://www.researchgate.net/publication/313732034_Monoaural_Audio_Source_Separation_Using_Deep_Convolutional_Neural_Networks>\
    
[2] E. Vincent, R. Gribonval, and C. Fevotte. Performance measurement in blind audio source separation. IEEE Transactions on Audio, Speech, and Language
    Processing, 2006. doi: 10.1109/TSA.2005.858005. Available: <https://ieeexplore.ieee.org/document/1643671>\
    
[3] Q. Kong, Y. Cao, H. Liu, K. Choi, and Y. Wang. Decoupling Magnitude and Phase Estimation with Deep ResUNet for Music Source Separation. ISMIR, 2021.
    arXiv:2109.05418. Available: <https://arxiv.org/abs/2109.05418>\
    
[4] K. M. Jeon, C. Chun, G. Kim, C. Leem, B. Kim and W. Choi. Lightweight U-Net Based Monaural Speech Source Separation for Edge Computing Device. ICCE, 2020.
    doi: 10.1109/ICCE46568.2020.9043051. Available: <https://ieeexplore.ieee.org/document/9043051>\
    
[5] K. J. Piczak. ESC: Dataset for Environmental Sound Classification. Proceedings of the 23rd Annual ACM Conference on Multimedia, 2015. Available:
    <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YDEPUT>\
    
[6] N. Turpault, R. Serizel, A. Parag Shah, and J. Salamon. Sound event detection in domestic environments with weakly labeled data and soundscape synthesis.
    Workshop on Detection and Classification of Acoustic Scenes and Events, Oct. 2019. Available: <https://project.inria.fr/desed/>\
    
[7] jiaaro. Pydub. Available: <https://pydub.com/>\

[8] S. Yao et al. STFNets: Learning Sensing Signals from the Time-Frequency Perspective with Short-Time Fourier Neural Networks. W3C2, 2019. arXiv:1902.07849. 
    Available: <https://arxiv.org/abs/1902.07849>\
    
[9] Arduino IDE. Available: <https://www.arduino.cc/en/software>\

[10] Jupyter Notebook. Available: <https://jupyter.org/>\

[11] Google Colab. Available: <https://colab.research.google.com/>\

# 7. Contributions

Justin Feng: Source separation ML models, dataset creation and preprocessing, website
Yang Liu: Arduino, microphone, and live testing

