import torch
from byol_pytorch import BYOL
from torchvision import models
import torch.utils.data as data
import medmnist
from medmnist import INFO
from torch.utils.data import Subset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Set up the argument parser
parser = argparse.ArgumentParser(description="Calculate the generalisation ability and diversity scores for a dataset")
parser.add_argument("-r", "--root_dir", type=str, help="Root directory where the code and data are located", default="/Users/katecevora/Documents/PhD")
parser.add_argument("-d", "--dataset", type=str, help="Dataset which we will use to run experiments", default="octmnist")
parser.add_argument("-i", "--image_size", type=int, help="Image size", default=28)

args = parser.parse_args()

# set up paths to directories
root_dir = args.root_dir
code_dir = os.path.join(root_dir, "code/DatasetDiversityMedMNIST")
output_dir = os.path.join(root_dir, "output")
data_dir = os.path.join(root_dir, "data/MedMNIST")
byol_dir = os.path.join(code_dir, "BYOL", "models")

image_size = args.image_size
batch_size = 20
n_epochs = 100
dataset = args.dataset
model_name = f"byol_{dataset}_{image_size}_weights.pth"
plot_name = f"byol_{dataset}_{image_size}_weights.png"
resnet = models.resnet50(pretrained=True)

learner = BYOL(
    resnet,
    image_size=image_size,
    hidden_layer='avgpool'
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

# DATA
info = INFO[dataset]
task = info['task']
DataClass = getattr(medmnist, info['python_class'])

data_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[.5], std=[.5])])

train_dataset = DataClass(split='train', transform=data_transform, download=False, as_rgb=True, size=image_size, root=data_dir)
ns = len(train_dataset)

if (dataset == "chestmnist") or (dataset == "octmnist"):
    print("sampling")
    data_loader = torch.utils.data.DataLoader(Subset(train_dataset, np.random.choice(ns, 6000)), batch_size=batch_size)
else:
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

loss_record = []
epochs = []

for epoch in range(n_epochs):
    epochs.append(epoch)
    loss_sum = 0
    print("Epoch {}".format(epoch))
    for image, _ in data_loader:
        loss = learner(image)
        loss_sum += image.size(0) * loss.item()  # scale loss by batch size
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder

    loss_record.append(loss_sum / len(data_loader.dataset))

    torch.save(learner.state_dict(), os.path.join(byol_dir, f"{epoch}_{model_name}"))

plt.clf()
plt.plot(epochs, loss_record)
plt.savefig(os.path.join(byol_dir, plot_name))



