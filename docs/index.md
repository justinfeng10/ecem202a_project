# Abstract

Audio recognition models can classify sounds individually, but can models classify and separate two or more sounds that are occurring at the same time, even on an embedded system? The goal of this project is to analyze audio source separation on the Arduino Nano 33 BLE Sense board. Source separation is the ability to take a mixture of 2 or more input target signals and to "separate" them into their component parts. In this project, the target signals are audio and the hope is to separate 2 target signals (e.g. faucet and microwave). 

The approach taken in this project is to first create a successful source separation model in TensorFlow. Next, audio input from the Arduino will be passed into the source separation model successfully. Once the model can successfully classify Arduino input, the next step is to deploy the TensorFlow model onto the Arduino via TensorFlow Lite and run live audio source separation on the Arduino.

# Team

* Justin Feng
* Yang Liu

# Required Submissions

* [Proposal](proposal.md)
* [Midterm Checkpoint Presentation Slides](http://)
* [Final Presentation Slides](http://)
* [Final Report](report)
