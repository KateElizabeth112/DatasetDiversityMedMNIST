# code to handle diversity scoring of a dataset of vectors
import numpy as np
import vendiScore
import torch
from torch.utils.data.dataset import Dataset
from copy import copy
from scipy.stats import entropy
from PIL import Image as im
from byolEncoder import BYOLEncoder
from samMedEncoder import SamMedEncoder


class DiversityScore:
    """
    Class for computing the diversity score for a dataset via a similarity matrix.
    """

    def __init__(self, data, data_rgb, indices, params):
        # check that the vectors parameter is a numpy array with two dimensions
        assert isinstance(params, dict), "params should be a dictionary"
        assert isinstance(data, Dataset), "train_data is not an instance of Dataset"

        self.params = params
        self.data = data
        self.data_rgb = data_rgb
        self.indices = indices
        self.code_dir = params["code_dir"]

    def __len__(self):
        return self.vectors.shape[0]

    def getPixelVectors(self, data):
        """
        Coverts a dataset of 2D images into an array of 1D pixel vectors
        :return:
        pixel_vectors: 2D pixel vector array
        """
        data_loader = torch.utils.data.DataLoader(data, batch_size=self.params["batch_size"], num_workers=self.params["n_workers"])

        for i, (images, labels) in enumerate(data_loader):
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

    def cosineSimilarity(self, vectorsA, vectorsB):
        """
        Compute cosine similarity between multiple vectors. Sets a class attribute.

        Returns:
        numpy.ndarray: Cosine similarity matrix.
        """

        # Compute dot product of vectors
        dot_product = np.dot(vectorsA, vectorsB.T)

        # Compute norms of vectors
        normA = np.linalg.norm(vectorsA, axis=1, keepdims=True)
        normB = np.linalg.norm(vectorsB, axis=1, keepdims=True)

        # Compute cosine similarity matrix
        similarity_matrix = dot_product / (normA * normB.T)

        return similarity_matrix

    def vendiScore(self, embed="pixel"):
        """
        Calculates the Vendi score directly from the cosine similarity matrix of pixel values.
        :return:
        float: The Vendi score for the dataset
        """
        if embed == "pixel":
            vectors = self.getPixelVectors(self.data)
        elif embed == "auto":
            encoder = BYOLEncoder(self.data_rgb, self.params)
            vectors = encoder.encode()
        elif embed == "inception":
            data = [im.fromarray(self.data[i][0].squeeze().numpy()) for i in range(len(self.data))]
            vectors = vendiScore.getInceptionEmbeddings(data)
        elif embed == "random":
            data = [im.fromarray(self.data[i][0].squeeze().numpy()) for i in range(len(self.data))]
            vectors = vendiScore.getInceptionEmbeddings(data, pretrained=False)
        elif embed == "sammed":
            encoder = SamMedEncoder(self.data, self.params)
            vectors = encoder.retrieve(self.indices)

        similarity_matrix = self.cosineSimilarity(vectors, vectors)

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

        if labels.shape[1] == 1:
            counts = np.zeros((2))
            counts[0] = np.sum(labels, axis=0)
            counts[1] = labels.shape[0] - np.sum(labels, axis=0)
        else:
            counts = np.sum(labels, axis=0)

        # calculate entropy from distribution over categorical labels
        label_entropy = entropy(counts)

        return label_entropy

    def domainGap(self, test_data):
        # turn training and test images into a stack of vectors
        train_vectors = self.getPixelVectors(self.data)
        test_vectors = self.getPixelVectors(test_data)

        # Compute cosine similarity matrix between train and test vectors
        similarity_matrix = self.cosineSimilarity(train_vectors, test_vectors)

        average_similarity = np.mean(similarity_matrix)

        return 1 - average_similarity

    def scoreDiversity(self):
        """
        Runs all diversity scoring methods, returns a dictionary of results.
        :return:
        """
        # Store the results in a dictionary
        results = {}
        results["label_entropy"] = self.labelEntropy()

        for embedding in ["pixel", "auto", "inception", "sammed", "random"]:
            vs, intdiv = self.vendiScore(embed=embedding)
            results["vs_{}".format(embedding)] = vs
            results["intdiv_{}".format(embedding)] = intdiv

        return results
