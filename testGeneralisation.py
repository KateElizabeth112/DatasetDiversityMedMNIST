# Takes a parameters file, runs a generalisation experiment and measures diversity of datastets
import argparse
import mlflow
import os
from torch.utils.data import Subset
from diversityScore import DiversityScore
import pickle as pkl
from datasetUtils import generateSubsetIndex, generateSubsetIndexDiverse
import numpy as np
from trainResNet import runTraining
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
import random

# Set up the argument parser
parser = argparse.ArgumentParser(description="Calculate the generalisation ability and diversity scores for a dataset")
parser.add_argument("-p", "--params_file", type=str, help="Name of params file.", default="local")
parser.add_argument("-r", "--root_dir", type=str, help="Root directory where the code and data are located",
                    default="/Users/katecevora/Documents/PhD")

args = parser.parse_args()

# set up paths to directories
root_dir = args.root_dir
code_dir = os.path.join(root_dir, "code/DatasetDiversityMedMNIST")
output_dir = os.path.join(root_dir, "output")
data_dir = os.path.join(root_dir, "data/MedMNIST")
params_file_path = os.path.join(code_dir, "params", args.params_file)
loss_plot_save_path = os.path.join(code_dir, "loss.png")

# define the transform to apply to the MedMNIST data
data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])

# Set our tracking server uri for logging with MLFlow if we are running locally
if args.params_file == "local":
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MedMNISTGeneralisation")

def main():
    # open a params file if specified
    if args.params_file != "local":
        assert os.path.exists(params_file_path), "The path {} to the params file does not exist.".format(
            params_file_path)

        f = open(params_file_path, "rb")
        params = pkl.load(f)
        f.close()
    else:
        # this local params file allows us to easily modify settings during development
        unique_id = "123456"
        params = {
            "data_category": "all",
            "n_samples": 200,
            "random_seed": 7,
            "n_layers": 3,
            "n_epochs": 3,
            "n_workers": 0,
            "batch_size": 20,
            "model_name": "classifier_{}.pt".format(unique_id),
            "dataset_name": "breastmnist",
            "diversity": "high",
            "image_size": 28,
            "code_dir": code_dir
        }

    # check the params file
    assert isinstance(params, dict), f"Expected a dictionary, but got {type(params).__name__}"

    # check what dataset we are using and load the data
    dataset_name = params["dataset_name"]
    image_size = params["image_size"]
    n_samples = params["n_samples"]
    diversity = params["diversity"]
    random_seed = params["random_seed"]
    n_epochs = params["n_epochs"]

    assert isinstance(dataset_name, str), "Dataset name must be a string."
    assert dataset_name in ["pneumoniamnist", "chestmnist", "breastmnist", "octmnist"], "The dataset name {} is not recognised."
    assert image_size in [28, 128, 224], "Image size {} is not an option. Must be one of 28, 128 or 224".format(
        image_size)

    print("Starting experiment with {0} dataset, image size {1}".format(dataset_name, image_size))

    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])

    # figure out what the data file is called
    if image_size == 28:
        data_file = f"{dataset_name}.npz"
    else:
        data_file = f"{dataset_name}_{image_size}.npz"

    # check whether we need to download the data
    if os.path.exists(os.path.join(data_dir, data_file)):
        download = False
    else:
        download = True

    train_data = DataClass(split='train', transform=data_transform, download=download, as_rgb=False, size=image_size,
                           root=data_dir)
    #val_data = DataClass(split='val', transform=data_transform, download=download, as_rgb=False, size=image_size,
    #                     root=data_dir)
    test_data = DataClass(split='test', transform=data_transform, download=download, as_rgb=False, size=image_size,
                          root=data_dir)

    train_data_rgb = DataClass(split='train',
                               transform=data_transform,
                               download=download,
                               as_rgb=True,
                               size=image_size,
                               root=data_dir)
    val_data_rgb = DataClass(split='val', transform=data_transform, download=download, as_rgb=True, size=image_size,
                         root=data_dir)
    test_data_rgb = DataClass(split='test', transform=data_transform, download=download, as_rgb=True, size=image_size,
                          root=data_dir)

    # if the dataset is chestmnist or octmnist, select a subset of the training and validation data since chestmnist is very large
    if dataset_name == "chestmnist" or dataset_name == "octmnist":
        random.seed(222)

        test_data_idx = random.sample(range(0, len(test_data)), 1000)
        test_data = Subset(test_data, test_data_idx)
        test_data_rgb = Subset(test_data_rgb, test_data_idx)

        val_data_idx = random.sample(range(0, len(val_data_rgb)), 1000)
        val_data_rgb = Subset(val_data_rgb, val_data_idx)

    print("Finished loading data.")

    # First randomly select a subset so that we don't have to compute a massive similarity matrix
    n_train_samples = len(train_data)
    idx_random_subset = generateSubsetIndex(train_data, "all", min(n_samples * 10, int(len(train_data) * 0.7)), random_seed)
    train_data_random_subset = Subset(train_data, idx_random_subset)

    # keep track of the original train data indices in the new subset so we can apply the selection to a new dataset
    # in one go
    idx_random_subset_orig = np.arange(0, n_train_samples)[idx_random_subset]

    # then choose maximally or minimally diverse samples from the training subset
    idx_diverse_subset = generateSubsetIndexDiverse(train_data_random_subset, "all", n_samples, diversity=diversity)

    # Find the indices of the diverse subset in the context of the original training data indices
    idx_diverse_subset_orig = idx_random_subset_orig[idx_diverse_subset]

    print(f"Finished sampling data. {idx_diverse_subset_orig.shape[0]} samples")

    print("Scoring data for diversity")
    
    # diversity score all datasets
    ds_train = DiversityScore(Subset(train_data, idx_diverse_subset), Subset(train_data_rgb, idx_diverse_subset_orig), np.array(idx_random_subset_orig), params)

    train_scores = ds_train.scoreDiversity()

    train_scores["domain_gap"] = ds_train.domainGap(test_data)

    print("Training ResNet for classification.")

    metrics = runTraining(Subset(train_data_rgb, idx_diverse_subset_orig),
                          val_data_rgb,
                          test_data_rgb,
                          dataset_name,
                          output_dir,
                          n_epochs,
                          5,
                          image_size,
                          'resnet18',
                          True)

    print("Finished experiment.")

    # record everything in MLFlow
    with mlflow.start_run():
        # Log the hyperparameters
        print("Starting mlflow logging")
        mlflow.log_params(params)

        # Log the diversity metrics for the training data
        split = "train"

        for score in ["vs", "intdiv"]:
            mlflow.log_metric(f"{score}_pixel_{split}", train_scores[f"{score}_pixel"])
            mlflow.log_metric(f"{score}_auto_{split}", train_scores[f"{score}_auto"])
            mlflow.log_metric(f"{score}_inception_{split}", train_scores[f"{score}_inception"])
            mlflow.log_metric(f"{score}_sammed_{split}", train_scores[f"{score}_sammed"])
            mlflow.log_metric(f"{score}_random_{split}", train_scores[f"{score}_random"])

        mlflow.log_metric(f"label_entropy_{split}", train_scores["label_entropy"])
        mlflow.log_metric("domain_gap", train_scores["domain_gap"])

        # log the metrics from training the classifier
        for metric in metrics.keys():
            mlflow.log_metric(metric, metrics[metric])

    print("Finished mlflow logging.")


if __name__ == "__main__":
    main()
