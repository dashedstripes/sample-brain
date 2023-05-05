import torch
from model import WaveNet
import numpy as np

model_weights_path = "/Users/adam/Code/sample-brain/model.pth"
model = WaveNet(10000)

def normalize_input(data, sequence_length=16000):
    if len(data) < sequence_length:
        padding = np.zeros(sequence_length - len(data))
        data = np.concatenate([data, padding])
    elif len(data) > sequence_length:
        data = data[: sequence_length]

    input = torch.tensor(data).float().unsqueeze(0).unsqueeze(0)
    input_normalized = (input - input.min()) / (input.max() - input.min())

    return input_normalized.clone().detach()


def test_model(model):
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()

    input_data = torch.randn(10000)
    normalized = normalize_input(input_data)

    output = model(normalized)

    print(output[-1])

test_model(model)