# namontiramo knjižnice
import torch
import torch.nn as nn
from torch import flatten
import sys
import os
from torch.utils.tensorboard import SummaryWriter
from models.U_net import Unet


# nastavitev hyperparametrov
num_epoch = 20
num_classes = 6
BATCH_SIZE = 64




# montiranje treninga na CUDO/CPU
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.manual_seed(42)
    print(f"CUDA is installed and ready", "YES" if torch.cuda.is_available() else "No")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Unet()
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epoch):
    model.train()
    epoch_losses = []
    epoch_accucacy = []

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        corrects = (predicted == labels).sum().item()
        accuracy = corrects / labels.size()

        epoch_losses.append(loss.item())
        epoch_accucacy.append(accuracy.item())

        if (i + 1) %100 == 0:
            print(f"Epoch = {i+1}, Epoch accuracy = {epoch_accucacy}")


print(f"Training finalized")





