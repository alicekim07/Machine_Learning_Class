# Import PyTorch
import torch
from torch import nn
from dataset import SpeechDataset
import requests
from pathlib import Path

# Download helper functions from Learn PyTorch repo
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download...")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)
# Import accuracy metric
from helper_functions import accuracy_fn

# Build a baseline model
class CNNKeyWordSpotter(nn.Module):
    def __init__(self, num_classes = 6):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))   
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model_baseline = CNNKeyWordSpotter(num_classes = 6)

# Setup Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_baseline.parameters(), lr = 0.001)

# Get Dataset
train_dataset = SpeechDataset("train_list.txt")
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = SpeechDataset("test_list.txt")
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

# Set the number of epochs
epochs = 3

# Create training and test loop
for epoch in range(epochs):
    print(f"Epoch: {epoch}\n------")
    ### Training
    train_loss = 0
    # Add a loop to loop through the training batches
    for batch, (X, y) in enumerate(train_dataloader):
        model_baseline.train()
        # 1. Forward pass
        y_pred = model_baseline(X)

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # Accumulate train loss

        # 3.Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Print out what's happening
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} Samples")
    
    # Devide Total train loss by length of train dataloader
    train_loss /= len(train_dataloader)

    ### Testing
    test_loss, test_acc = 0, 0
    model_baseline.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            # 1. Forward pass
            test_pred = model_baseline(X_test)

            # 2. Calculate loss (accumulatively)
            test_loss += loss_fn(test_pred, y_test)

            # 3. Calculate accuracy (Accumulatively)
            test_acc += accuracy_fn(y_true = y_test, y_pred = test_pred.argmax(dim = 1))

        # Calculate the test loss average per batch
        test_loss /= len(test_dataloader)

        # Calculate the test acc average per batch
        test_acc /= len(test_dataloader)

        # Print out what's happening
        print(f"\nTrain loss: {train_loss:.5f} | Test loss : {test_loss:.5f}, Test acc : {test_acc:.2f}")
