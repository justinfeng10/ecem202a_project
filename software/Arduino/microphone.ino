/*
  Inspired by Arduino microphone example
*/

#include <PDM.h>


// buffer to read samples into, each sample is 16-bits
short sampleBuffer[16000];

// number of samples read
volatile int samplesRead;
int incomingByte;
void setup() {
  Serial.begin(9600);
  while (!Serial);

  // configure the data receive callback
  PDM.onReceive(onPDMdata);

  // initialize PDM with:
  // - one channel (mono mode)
  // - a 16 kHz sample rate
  PDM.setBufferSize(32000);
  
  if (!PDM.begin(1, 16000)) {
    Serial.println("Failed to start PDM!");
    while (1);
  }
  
  // Set gain for sufficient volume
  PDM.setGain(100);
}


void loop() {
  // wait for samples to be read 
  if (samplesRead) {
    incomingByte=Serial.read();
    // If receive user input, print data
    if(incomingByte == 'v'){
      // print samples to the serial monitor or plotter
      for (int i = 0; i < samplesRead; i++) {
        Serial.println(sampleBuffer[i]);
      }

      // clear the read count
      samplesRead = 0;
    }
  }
}
  
 
void onPDMdata() {
  // query the number of bytes available
  
  int bytesAvailable = PDM.available();
  
  // read into the sample buffer
  PDM.read(sampleBuffer, bytesAvailable);

  // 16-bit, 2 bytes per sample
  samplesRead = bytesAvailable / 2;

}
