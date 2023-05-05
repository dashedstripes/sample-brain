import torch.nn as nn
from model import WaveNet
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import WaveNetDataset

directory_path = '/Users/adam/Code/sample-brain/data'
dataset = WaveNetDataset(directory_path)
model = WaveNet(dataset[0][0].shape[-1])

# wavenet predicts a SINGLE SAMPLE at a time
# in order to generate a single second of audio, you'd need to run this 44100 times!!!

def train_model(model, dataset, num_epochs=5, batch_size=1, learning_rate=0.001):
    # Set up the data loader and optimizer
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Set up the loss function
    criterion = nn.MSELoss()
    
    # Train the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (input, target) in enumerate(dataloader):
            inputs = input[:, :, 1:]
            targets = target[:, :, :1]

            # # Zero the parameter gradients
            optimizer.zero_grad()

            # # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # Print every 100 mini-batches
                print(f"Epoch {epoch + 1}, Batch {i + 1}: Loss = {running_loss / 100:.6f}")
                running_loss = 0.0
                
    print("Finished training!")


train_model(model, dataset)