import torch
import os
import torchvision
from torchvision import transforms
import torch.utils.data as data

BATCH_SIZE = 64

TRAIN_DATA_PATH = "laboro_tomato/train/images"
TEST_DATA_PATH = "laboro_tomato/test/images"

"""transform_img = transforms([



])"""

train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH)
train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH)
test_data_loader  = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) 

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=TRAIN_DATA_PATH,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=TEST_DATA_PATH,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)
