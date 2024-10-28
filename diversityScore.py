# code to handle diversity scoring of a dataset of vectors
import numpy as np
import vendiScore
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from copy import copy
from scipy.stats import entropy
from PIL import Image as im
from autoencoder2D import ConvAutoencoder


class DiversityScore:
    """
    Class for computing the diversity score for a dataset via a similarity matrix.
    """

    def __init__(self, data, params):
        # check that the vectors parameter is a numpy array with two dimensions
        assert isinstance(params, dict), "params should be a dictionary"
        assert isinstance(data, Dataset), "train_data is not an instance of Dataset"

        self.params = params
        self.data = data

        # set up a data loader
        self.data_loader = torch.utils.data.DataLoader(self.data, batch_size=self.params["batch_size"],
                                                       num_workers=self.params["n_workers"])

    def __len__(self):
        return self.vectors.shape[0]

    def getEmbeddings(self):
        """
        Retrieves the compressed representations of a dataset from the middle layer of an autoencoder. Loads the most
        recent model checkpoint to compute representations.
        :return:
        vectors numpy.ndarray: 2D array of vector representations of each image.
        """
        # figure out which model to load depending on the dataset and the image size
        # TODO
        save_path = "/Users/katecevora/Documents/PhD/code/AutoencoderMNIST/models/autoencoderMedMNISTfull.pt"
        model_name = "autoencoder_{0}_{1}.pt".format(self.params["dataset_name"], self.params["image_size"])

        # load the model from the latest checkpoint
        checkpoint = torch.load(save_path)
        model = ConvAutoencoder(save_path=save_path)

        # Load the model state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])

        # iterate over all images in the dataset and extract representations
        for i, (images, labels) in enumerate(self.data_loader):
            preds, images_compressed = model(images)

            # convert to a numpy array for easier handling after detaching gradient
            images_compressed = images_compressed.detach().numpy()

            # flatten all dimensions except the first
            flattened_size = np.prod(images_compressed.shape[1:])
            images_compressed = images_compressed.reshape((images_compressed.shape[0], flattened_size))

            # stack the results from each batch until we have run over the entire dataset
            if i == 0:
                # assign the values of images compressed to output
                vectors = copy(images_compressed)
            else:
                vectors = np.vstack((vectors, images_compressed))

        assert len(vectors.shape) == 2, "The output array should have two dimensions but it has {}".format(
            len(vectors.shape))

        return vectors

    def getPixelVectors(self):
        """
        Coverts a dataset of 2D images into an array of 1D pixel vectors
        :return:
        pixel_vectors: 2D pixel vector array
        """
        for i, (images, labels) in enumerate(self.data_loader):

            # convert to a numpy array for easier handling after detaching gradient
            images = images.numpy()

            # flatten all dimensions except the first
            flattened_size = np.prod(images.shape[1:])
            images_flat = images.reshape((images.shape[0], flattened_size))

            # stack the results from each batch until we have run over the entire dataset
            if i == 0:
                # assign the values of images compressed to output
                pixel_vectors = copy(images_flat)
            else:
                pixel_vectors = np.vstack((pixel_vectors, images_flat))

        assert len(pixel_vectors.shape) == 2, "The output array should have two dimensions but it has {}".format(
            len(pixel_vectors.shape))

        return pixel_vectors

    def cosineSimilarity(self, vectors):
        """
        Compute cosine similarity between multiple vectors. Sets a class attribute.

        Returns:
        numpy.ndarray: Cosine similarity matrix.
        """

        # Compute dot product of vectors
        dot_product = np.dot(vectors, vectors.T)

        # Compute norms of vectors
        norm = np.linalg.norm(vectors, axis=1)
        norm = norm[:, np.newaxis]  # Convert to column vector

        # Compute cosine similarity
        similarity_matrix = dot_product / (norm * norm.T)

        return similarity_matrix

    def vendiScore(self, embed="pixel"):
        """
        Calculates the Vendi score directly from the cosine similarity matrix of pixel values.
        :return:
        float: The Vendi score for the dataset
        """
        if embed == "pixel":
            vectors = self.getPixelVectors()
        elif embed == "auto":
            vectors = self.getEmbeddings()
        elif embed == "inception":
            data = [im.fromarray(self.data[i][0].squeeze().numpy()) for i in range(len(self.data))]
            vectors = vendiScore.getInceptionEmbeddings(data)

        similarity_matrix = self.cosineSimilarity(vectors)

        score = vendiScore.score_K(similarity_matrix)

        intdiv = vendiScore.intdiv_K(similarity_matrix)

        return score, intdiv

    def labelEntropy(self):
        """
        calculate the entropy of the labels in the dataset
        :return:
        """
        # get the dataset labels
        labels = np.array([self.data[i][1] for i in range(len(self.data))])

        # convert the labels into a distribution
        categories = np.unique(labels)
        counts = np.zeros(categories.shape[0])

        for i in range(len(categories)):
            counts[i] = np.sum(labels == categories[i])

        # calculate entropy from distribution over categorical labels
        label_entropy = entropy(counts)

        return label_entropy

    def scoreDiversity(self):
        """
        Runs all diversity scoring methods, returns a dictionary of results.
        :return:
        """
        # Store the results in a dictionary
        results = {}
        for embedding in ["pixel", "auto", "inception"]:
            vs, intdiv = self.vendiScore(embed=embedding)
            results["vs_{}".format(embedding)] = vs
            results["intdiv_{}".format(embedding)] = intdiv

        results["label_entropy"] = self.labelEntropy()

        return results
