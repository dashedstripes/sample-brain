SAMPLE BRAIN

Any sample. Instantly.

## Goal

This model will be simple. It will be trained to understand electronic snare samples. It will be able to generate a new snare sample.

# Background

This is build using WaveNet architecture. WaveNet takes in an audio file and generates what it thinks the next sample in the audio would be. In order to generate a full audio sample from a trained WaveNet model, you'd need to begin with random noise, then repeatedly pass the output of the model back in as input, do this 44100 times per second, and you'll have a unique, generated piece of audio!

# Set up

1. Add .wav files to the `/data` folder
2. Run train.py

# Issues

Read the code, then fix it lol