import torch
from model import WaveNet
import numpy as np
import wave

model_weights_path = "/Users/adam/Code/sample-brain/model.pth"
sequence_length = 16000
model = WaveNet(sequence_length)


def normalize_input(data, sequence_length=sequence_length):
    if len(data) < sequence_length:
        padding = np.zeros(sequence_length - len(data))
        data = np.concatenate([data, padding])
    elif len(data) > sequence_length:
        data = data[: sequence_length]

    input = data.clone().detach().float().unsqueeze(0).unsqueeze(0)
    input_normalized = (input - input.min()) / (input.max() - input.min())

    return input_normalized.clone().detach()

def write_wav_file(data):
  sample_rate = 44100  # Hz
  num_channels = 1  # mono
  sample_width = 2  # bytes

  with wave.open('output.wav', 'wb') as wav_file:
    wav_file.setnchannels(num_channels)
    wav_file.setsampwidth(sample_width)
    wav_file.setframerate(sample_rate)

    # Write the audio data to the file
    wav_file.writeframes(data.tobytes())

def generate_sound(model, length=sequence_length):
    with torch.no_grad():
      model.load_state_dict(torch.load(model_weights_path))
      model.eval()

      data = torch.randn(length)
      data = normalize_input(data)

      for i in range(10):  
        output = model(data)
        data = torch.roll(data, -1, 2)
        data[-1][-1] = output

    write_wav_file(data[-1][-1].numpy())

def test_model(model):
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()

    input_data = torch.randn(sequence_length)
    normalized = normalize_input(input_data)

    output = model(normalized)

    print(output[-1])

generate_sound(model, sequence_length)