# Takes a parameters file, runs a generalisation experiment and measures diversity of datastets
import argparse
import mlflow
import os
from torch.utils.data import Subset
from diversityScore import DiversityScore
import pickle as pkl
from datasetUtils import generateSubsetIndex, generateSubsetIndexDiverse, RotationTransform
import numpy as np
from trainResNet import runTraining
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO

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
            "dataset_name": "pneumoniamnist",
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
    download = False # Flag to indicate whether we should download the data

    assert isinstance(dataset_name, str), "Dataset name must be a string."
    assert dataset_name in ["pneumoniamnist", "chestmnist"], "The dataset name {} is not recognised."
    assert image_size in [28, 128, 224], "Image size {} is not an option. Must be one of 28, 128 or 224".format(
        image_size)

    print("Starting experiment with {0} dataset, image size {1}".format(dataset_name, image_size))

    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])

    train_data = DataClass(split='train', transform=data_transform, download=download, as_rgb=False, size=image_size,
                           root=data_dir)
    val_data = DataClass(split='val', transform=data_transform, download=download, as_rgb=False, size=image_size,
                         root=data_dir)
    test_data = DataClass(split='test', transform=data_transform, download=download, as_rgb=False, size=image_size,
                          root=data_dir)

    train_data_rgb = DataClass(split='train',
                               transform=data_transform,
                               download=download,
                               as_rgb=True,
                               size=image_size)
    val_data_rgb = DataClass(split='val', transform=data_transform, download=download, as_rgb=True, size=image_size,
                         root=data_dir)
    test_data_rgb = DataClass(split='test', transform=data_transform, download=download, as_rgb=True, size=image_size,
                          root=data_dir)

    print("Finished loading data.")

    # First randomly select a subset so that we don't have to compute a massive similarity matrix
    n_train_samples = len(train_data)
    idx_train = generateSubsetIndex(train_data, "all", min(n_samples * 5, len(train_data)), random_seed)
    train_data = Subset(train_data, idx_train)

    # keep track of the original train data indices in the new subset so we can apply the selection to a new dataset
    # in one go
    idx_train_orig = np.arange(0, n_train_samples)
    idx_train_mod = idx_train_orig[idx_train]

    # then choose maximally or minimally diverse samples from the training subset
    subset_idx = generateSubsetIndexDiverse(train_data, "all", n_samples, diversity=diversity)
    idx_train_final = idx_train_mod[subset_idx]

    print("Finished sampling data. {} samples".format(idx_train_final.shape[0]))

    print("Scoring data for diversity")

    # diversity score all datasets
    ds_train = DiversityScore(Subset(train_data, subset_idx), Subset(train_data_rgb, idx_train_final), np.array(idx_train_final), params)

    train_scores = ds_train.scoreDiversity()

    print("Training ResNet for classification.")

    metrics = runTraining(Subset(train_data_rgb, idx_train_final),
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
        ds = "train"

        for s in ["vs", "intdiv"]:
            mlflow.log_metric("{0}_pixel_{1}".format(s, ds), train_scores["{}_pixel".format(s)])
            mlflow.log_metric("{0}_auto_{1}".format(s, ds), train_scores["{}_auto".format(s)])
            mlflow.log_metric("{0}_inception_{1}".format(s, ds), train_scores["{}_inception".format(s)])
            mlflow.log_metric("{0}_sammed_{1}".format(s, ds), train_scores["{}_sammed".format(s)])
            mlflow.log_metric("{0}_random_{1}".format(s, ds), train_scores["{}_random".format(s)])

        mlflow.log_metric("label_entropy_{}".format(ds), train_scores["label_entropy"])

        # log the metrics from training the classifier
        for metric in metrics.keys():
            mlflow.log_metric(metric, metrics[metric])

    print("Finished mlflow logging.")


if __name__ == "__main__":
    main()
