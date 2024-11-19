# iterate across a dataset and save the SAMMed encodings
import argparse
from samMedEncoder import SamMedEncoder
import medmnist
from medmnist import INFO
import os
import torchvision.transforms as transforms

# Set up the argument parser
parser = argparse.ArgumentParser(description="Precompute and store SAMMed Encodings for a dataset")
parser.add_argument("-r", "--root_dir", type=str, help="Root directory where the code and data are located",
                    default="/Users/katecevora/Documents/PhD")
parser.add_argument("-d", "--dataset_name", type=str, help="Name of dataset.", default="pneumoniamnist")
parser.add_argument("-i", "--image_size", type=int, help="Size of the images", default=28)

args = parser.parse_args()

root_dir = args.root_dir
code_dir = os.path.join(root_dir, "code/DatasetDiversityMedMNIST")
output_dir = os.path.join(root_dir, "output")
data_dir = os.path.join(root_dir, "data/MedMNIST")

dataset_name = args.dataset_name
image_size = args.image_size

info = INFO[dataset_name]
DataClass = getattr(medmnist, info['python_class'])

# define the transform to apply to the MedMNIST data
data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])

data = DataClass(split='train', transform=data_transform, download=False, as_rgb=False, size=image_size, root=data_dir)

print(len(data))

params = {"code_dir": code_dir,
          "dataset_name": dataset_name,
          "image_size": image_size}

encoder = SamMedEncoder(data, params)
encoder.encode()
