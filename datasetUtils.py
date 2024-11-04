# Contains functions for handling datasets
import numpy as np
import torch
import random
import torchvision.transforms as transforms
from diversityScore import DiversityScore
from torch.utils.data import Subset


def getSimilarSamples(data, n_samples, diversity=None):
    assert diversity is not None, "Please select the order for the similarity ranking, high or low"
    assert diversity in ["high", "low", "random"], "Please choose either high, low or random for the similarity ranking order"

    if diversity == "random":
        return np.array(random.sample(range(0, len(data)), n_samples))
    else:
        # create a diversity scoring object using the full dataset
        params = {"n_epochs": 10,
                  "n_workers": 0,
                  "batch_size": 2}

        ds = DiversityScore(data, None, params)

        # get the similarity matrix for the whole dataset
        vectors = ds.getPixelVectors()
        sim_matrix = ds.cosineSimilarity(vectors)

        # find average similarity for each sample
        average_sim = np.mean(sim_matrix, axis=0)

        # order the average similarity vector (smallest first) and find the first N samples
        if diversity == "low":
            index = np.flip(np.argsort(average_sim))[:n_samples]
        elif diversity == "high":
            index = np.argsort(average_sim)[:n_samples]

        return index

def generateSubsetIndexDiverse(data, category, n_samples, diversity="None"):
    """
    Generate an index of data samples to use that are maximally or minimally diverse
    :param data:
    :param category:
    :param n_samples:
    :param random_seed:
    :return:
    """
    assert diversity is not None, "Please select the order for the similarity ranking, high or low"
    assert diversity in ["high", "low"], "Please choose either high or low for the similarity ranking order"
    assert isinstance(n_samples, int), "The number of samples must be an integer"

    if category == "all":
        subset_idx = getSimilarSamples(data, n_samples, diversity=diversity)

    else:
        assert isinstance(category, int), "The data category {} must be equal to the string \"all\" or an integer."
        assert category <= 9, "The category value cannot be greater than 9."
        assert category >= 0, "The category value cannot be less than 0"

        # create a data loader
        dataset_loader = torch.utils.data.DataLoader(data, batch_size=20, num_workers=0)

        # iterate over the dataset to get the labels
        labels = []
        for _, label in dataset_loader:
            labels += list(label.numpy())

        # convert to numpy array
        labels = np.array(labels)

        # get an index of locations where the label is equal to the value of category
        idx_cat = np.where(labels == category)[0]

        # check that the number of samples is less than the number of datapoints in that category
        assert n_samples <= idx_cat.shape[
            0], "The number of samples ({}) must be less than the number of datapoints in the category ({})".format(
            n_samples, idx_cat.shape[0])

        # Create a subset of samples for a single category
        data_cat = Subset(data, idx_cat)

        # Find the most similar (or dissimilar) samples
        idx_sim = getSimilarSamples(data_cat, n_samples, diversity=diversity)

        # convert the within-category index to a full dataset index
        subset_idx = idx_cat[idx_sim]

    assert subset_idx.shape[0] == n_samples, "The number of samples in the idx_sample array does not match n_samples"

    return subset_idx


def generateSubsetIndex(data, category, n_samples, random_seed, train=True):
    # generate an index of data samples to use
    assert isinstance(n_samples, int), "The number of samples must be an integer"

    # TODO why doesn't this work here but it's fine in the main function
    # open the full dataset
    #data = datasets.MNIST(root=dataset_root, train=train, download=False, transform=transforms.ToTensor()),

    if category == "all":
        subset_idx = np.array(random.sample(range(0, len(data)), n_samples))

    else:
        assert isinstance(category, int), "The data category {} must be equal to the string \"all\" or an integer."
        assert category <= 9, "The category value cannot be greater than 9."
        assert category >= 0, "The category value cannot be less than 0"

        # create a data loader
        dataset_loader = torch.utils.data.DataLoader(data, batch_size=20, num_workers=0)

        # iterate over the dataset to get the labels
        labels = []
        for data in dataset_loader:
            _, label = data
            labels += list(label.numpy())

        # convert to numpy array
        labels = np.array(labels)

        # get an index of locations where the label is equal to the value of category
        idx = np.where(labels == category)[0]

        # check that the number of samples is less than the number of datapoints in that category
        assert n_samples <= idx.shape[
            0], "The number of samples ({}) must be less than the number of datapoints in the category ({})".format(
            n_samples, idx.shape[0])

        # sample with a random seed
        random.seed(random_seed)
        random_idx = random.sample(range(0, idx.shape[0]), n_samples)

        subset_idx = idx[random_idx]

    assert subset_idx.shape[0] == n_samples, "The number of samples in the idx_sample array does not match n_samples"

    return subset_idx

class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        x = transforms.functional.hflip(x)
        x = transforms.functional.rotate(x, self.angle)
        return x


