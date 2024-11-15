# iterate across a dataset and save the SAMMed encodings
from samMedEncoder import SamMedEncoder
import medmnist
from medmnist import INFO
import os
import torchvision.transforms as transforms
import pickle as pkl

root_dir = "/Users/katecevora/Documents/PhD"
code_dir = os.path.join(root_dir, "code/DatasetDiversityMedMNIST")
output_dir = os.path.join(root_dir, "output")
data_dir = os.path.join(root_dir, "data/MedMNIST")

dataset_name = "pneumoniamnist"
image_size = 28

info = INFO[dataset_name]
DataClass = getattr(medmnist, info['python_class'])

# define the transform to apply to the MedMNIST data
data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])

data = DataClass(split='train', transform=data_transform, download=False, as_rgb=False, size=image_size, root=data_dir)

print(len(data))

encoder = SamMedEncoder(data, {})
encoder.encode()
