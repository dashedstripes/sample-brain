import numpy as np
import librosa
import os

# Set the directory path where the WAV files are stored
directory_path = "/Users/adam/Code/sample-brain/data"

# Set the desired length of the audio clips
clip_length = 1 # in seconds

# Set the sample rate for the audio clips
sample_rate = 16000

# Loop through each WAV file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".wav"):
        
        # Load the WAV file
        wav_path = os.path.join(directory_path, filename)
        audio, sr = librosa.load(wav_path, sr=sample_rate, mono=True)
        
        # Split the audio into clips of the desired length
        clip_samples = clip_length * sample_rate
        num_clips = len(audio) // clip_samples
        audio = audio[:num_clips * clip_samples]  # truncate audio to fit evenly into clips
        audio = audio.reshape(num_clips, clip_samples) 

        # Save each clip as a separate file
        for i in range(num_clips):
            clip = audio[i]
            clip_filename = filename.replace(".wav", f"_{i}.wav")
            clip_path = os.path.join(directory_path, clip_filename)
            librosa.output.write_wav(clip_path, clip, sr=sr)
            
print("Done!")
