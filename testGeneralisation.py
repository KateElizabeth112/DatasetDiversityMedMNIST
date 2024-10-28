# Takes a parameters file, runs a generalisation experiment and measures diversity of datastets
import argparse
import mlflow
import os
import torchvision.transforms as transforms
#from trainers import Trainer, TrainerParams, Inference
#from autoencoder2D import ConvAutoencoder
#from classifier import ImageClassifier
from torch.utils.data import Subset
from torchvision import datasets
#from diversityScore import DiversityScore
import pickle as pkl
from datasetUtils import generateSubsetIndex, generateSubsetIndexDiverse, RotationTransform
#import plotting
import numpy as np
from medMNISTDataset import MedNISTDataset
from trainResNet import runTraining

# Set up the argument parser
parser = argparse.ArgumentParser(description="Calculate the generalisation ability and diversity scores for a dataset")
parser.add_argument("-e", "--experiment", type=str, help="Name of the experiment.", default="GeneralisationMinMaxDiversity")
parser.add_argument("-p", "--params_file", type=str, help="Name of params file.", default="test_params.pkl")
parser.add_argument("-r", "--root_dir", type=str, help="Root directory where the code and data are located", default="/Users/katecevora/Documents/PhD")

args = parser.parse_args()

root_dir = args.root_dir
code_dir = os.path.join(root_dir, "code/AutoencoderMNIST")
data_dir = os.path.join(root_dir, "data")
experiment_name = args.experiment

assert experiment_name in ["Generalisation_Fixed_Entropy", "GeneralisationMinMaxDiversity"], "Experiment name is not recognised"

params_file_path = os.path.join(code_dir, "params", experiment_name, args.params_file)
models_path = os.path.join(code_dir, "models")
model_save_path = os.path.join(code_dir, "models")
loss_plot_save_path = os.path.join(code_dir, "loss.png")

# number of test/valid dataset samples per category
number_test_samples_per_cat = 500

# convert data to torch.FloatTensor
transform_mnist = transforms.ToTensor()
transform_emnist = transforms.Compose([transforms.ToTensor(), RotationTransform(-270)])

# Set our tracking server uri for logging with MLFlow
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
#mlflow.set_experiment(experiment_name)


def getDataSubsets(data, n_samples, diversity="high"):
    subset_idx = generateSubsetIndexDiverse(data, "all", n_samples, diversity=diversity)

    return Subset(data, subset_idx)

def main():
    # open a params file
    assert os.path.exists(params_file_path), "The path {} to the params file does not exist.".format(params_file_path)

    f = open(params_file_path, "rb")
    params = pkl.load(f)
    f.close()

    # check the params file
    assert isinstance(params, dict), f"Expected a dictionary, but got {type(params).__name__}"

    # check what dataset we are using and load the data
    dataset_name = params["dataset_name"]
    assert isinstance(dataset_name, str), "Dataset name must be a string."
    assert dataset_name in ["MNIST", "EMNIST", "PneuNIST"], "The dataset name {} is not recognised."

    print("Starting TestGeneralisation {0} experiment with {1} dataset".format(experiment_name, dataset_name))

    train_data = MedNISTDataset(data_dir, split="train", task="pneumoniamnist")
    valid_data = MedNISTDataset(data_dir, split="val", task="pneumoniamnist")
    test_data = MedNISTDataset(data_dir, split="test", task="pneumoniamnist")
    out_features = 2
    ae_model_name = "autoencoderMedMNISTfull.pt"

    print("Finished loading data.")

    if len(train_data) < params["n_samples"] * 5:
        print("Warning: train data has {0} samples not {1}".format(len(train_data), params["n_samples"] * 5))
    if len(valid_data) < number_test_samples_per_cat * 10:
        print("Warning: validation data has {0} samples not {1}".format(len(valid_data), number_test_samples_per_cat * 10))
    if len(test_data) < number_test_samples_per_cat * 10:
        print("Warning: test data has {0} samples not {1}".format(len(test_data), number_test_samples_per_cat * 10))

    # First randomly select a subset so that we don't have to compute a massive similarity matrix
    n_train_samples = len(train_data)
    idx_train = generateSubsetIndex(train_data, "all", min(params["n_samples"] * 5, len(train_data)), params["random_seed"])

    train_data = Subset(train_data, idx_train)

    # keep track of the original train data indices in the new subset so we can apply the selection to a new dataset
    # in one go
    idx_train_orig = np.arange(0, n_train_samples)
    idx_train_mod = idx_train_orig[idx_train]

    # then choose maximally or minimally diverse samples from the training subset
    subset_idx = generateSubsetIndexDiverse(train_data, "all", params["n_samples"], diversity=params["diversity"])

    idx_train_final = idx_train_mod[subset_idx]
    print("Finished sampling data.")

    """
    # load the AE model that we will use to embed the data
    model_ae = ConvAutoencoder(save_path=os.path.join(models_path, ae_model_name))

    trainer_params = TrainerParams(n_epochs=params["n_epochs"], num_workers=params["n_workers"],
                                   batch_size=params["batch_size"])

    # diversity score all datasets
    ds_train = DiversityScore(train_data, trainer_params, model_ae)
    #ds_test = DiversityScore(test_data, trainer_params, model_ae)
    #ds_valid = DiversityScore(valid_data, trainer_params, model_ae)

    train_scores = ds_train.scoreDiversity()
    #test_scores = ds_test.scoreDiversity()
    #valid_scores = ds_valid.scoreDiversity()
    """
    print("Training ResNet for classification.")

    runTraining(idx_train_final,
                'pneumoniamnist',
                './output',
                3,
                '0',
                50,
                28,
                True,
                'resnet18',
                True,
                True,
                None,
                'model1')

    print("Finished experiment.")

    # record everything in MLFlow
    with mlflow.start_run():
        # Log the hyperparameters
        print("Starting mlflow logging")
        mlflow.log_params(params)

        # Log the diversity metrics
        for (ds, scores) in zip(["train", "test", "valid"], [train_scores, test_scores, valid_scores]):
            for s in ["vs", "av_sim", "intdiv"]:
                mlflow.log_metric("{0}_pixel_{1}".format(s, ds), scores["{}_pixel".format(s)])
                mlflow.log_metric("{0}_embed_full_{1}".format(s, ds), scores["{}_auto".format(s)])
                mlflow.log_metric("{0}_inception_{1}".format(s, ds), scores["{}_inception".format(s)])

            mlflow.log_metric("vs_entropy_{}".format(ds), scores["label_entropy"])

        # log the generalisation accuracy
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("valid_accuracy", valid_accuracy)

        # log the loss plot for the classification model
        #mlflow.log_artifact(loss_plot_save_path)

    print("Finished mlflow logging.")


if __name__ == "__main__":
    main()
