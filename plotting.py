# tools to plot the results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr, ConstantInputWarning
import warnings

warnings.simplefilter("ignore", ConstantInputWarning)

lblu = "#add9f4"
lred = "#f36860"
lgrn = "#7dda7e"


class ResultsProcesser:
    """
    Class for plotting results of generalisation experiments.
    """
    def __init__(self, experiment_name=""):
        # Check the types of the arguments are as expected
        assert isinstance(experiment_name, str), "The experiment name must be a string"
        assert os.path.exists(os.path.join("./results", experiment_name + ".csv")), "Path to results CSV does not exist."

        self.experiment_name = experiment_name
        self.csv_path = os.path.join("./results", experiment_name + ".csv")
        self.results = pd.read_csv(self.csv_path)

        # create a list of diversity score metrics
        score_titles = ["VS", "IntDiv"]
        scores = ["vs", "intdiv"]
        embed_titles = [" (Raw Pixel)", " (BYOL)", " (Inception)", " (Random)", " (SAMMed)"]
        embed = ["pixel", "auto", "inception", "random", "sammed"]

        plot_titles = []
        diversity_scores = []

        for k in range(len(embed)):
            for j in range(len(scores)):
                plot_titles.append(score_titles[j] + embed_titles[k])
                diversity_scores.append("{0}_{1}_train".format(scores[j], embed[k]))

        # Add number of samples and label entropy
        plot_titles.append("Label Entropy")
        diversity_scores.append("label_entropy_train")

        self.diversity_scores = diversity_scores
        self.plot_titles = plot_titles

    def plot(self, output="test_acc", dataset="", image_size=28):
        assert isinstance(dataset, list), "Please specify the dataset/s within a list"

        # Check that the specified dataset/s are present in the results file
        for ds in dataset:
            assert ds in list(np.unique(self.results["dataset_name"].values)), "Dataset {} not found in results".format(ds)

        fig, axes = plt.subplots(nrows=4, ncols=3, sharey=True)

        # list of colours to use for plotting different datasets
        colours_list = [lred, lblu, lgrn]

        # if a dataset was not specified, get a list of datasets from the results
        if dataset == "":
            dataset_names = list(np.unique(self.results["dataset_name"].values))
        else:
            dataset_names = list(dataset)

        num_datasets = len(dataset_names)
        colours = colours_list[:num_datasets]

        for c, ds in zip(colours, dataset_names):

            # check how many different sample sizes we have for this dataset and image size combo
            condition1 = self.results["dataset_name"] == ds
            condition2 = self.results["image_size"] == image_size
            n_samples = np.unique(self.results["n_samples"][condition1 & condition2].values)
            #diversity_types = np.unique(self.results["diversity"][condition1 & condition2].values)
            diversity_types = ["high", "random", "low"]
            diversity_colours = ["r", "g", "b"]

            # generate some colours
            colours = colours_list[:len(n_samples)]

            for i in range(len(self.diversity_scores)):
                # Check that we have this column present in the results CSV, if not, just skip
                if self.diversity_scores[i] in self.results.columns:
                    ax = axes.flat[i]

                    # iterate over the number of samples in the training dataset
                    for c, ns in zip(colours, n_samples):
                        for d, dc in zip(diversity_types, diversity_colours):
                            # filter by the diversity metric, dataset name and the number of samples in training data
                            condition1 = self.results["dataset_name"] == ds
                            condition2 = self.results["n_samples"] == ns
                            condition3 = self.results["image_size"] == image_size
                            condition4 = self.results["diversity"] == d

                            condition = condition1 & condition2 & condition3 & condition4

                            diversity = self.results[self.diversity_scores[i]][condition].values

                            if output in ["test_AUC", "val_AUC", "test_acc", "val_acc"]:
                                accuracy = self.results[output][condition].values
                            elif output == "generalisation_gap":
                                valid_accuracy = self.results["val_AUC"][condition].values
                                test_accuracy = self.results["test_AUC"][condition].values
                                accuracy = test_accuracy - valid_accuracy / (0.5 * (test_accuracy + valid_accuracy))
                            else:
                                print("Metric {} not recognised".format(output))

                            # Find out if we have any Nan values in scores (due to missing data)
                            nan_idx = np.isnan(diversity)

                            # filter out nan entries
                            diversity_nonan = diversity[np.invert(nan_idx)]
                            accuracy_nonnan = accuracy[np.invert(nan_idx)]

                            # Check whether we have any data for this metric
                            if diversity_nonan.shape[0] > 0:
                                # calculate the correlation coefficient (returns an object)
                                ax.scatter(diversity_nonan, accuracy_nonnan, color=dc, label="n_samples={0}".format(ns))
                        ax.set_xlabel(self.plot_titles[i])

        # Turn off the last plot's axes
        ax = axes.flat[i+1]
        for c, ns in zip(colours, n_samples):
            ax.scatter([], [], color=c, label="n_samples={0}".format(ns))
        ax.axis("off")
        ax.legend()

        fig.text(0.015, 0.5, 'Test Set Accuracy', ha='center', va='center', rotation='vertical')
        plt.tight_layout()
        plt.show()

    def __printCorrelation__(self, diversity, accuracy):
        """
        Helper function for calculating and printing the correlation between diversity scores and test accuracy.
        :param diversity:
        :param accuracy:
        :return:
        """
        # filter out the nans
        nan_idx = np.isnan(diversity)
        diversity_nonan = diversity[np.invert(nan_idx)]
        accuracy_nonnan = accuracy[np.invert(nan_idx)]

        # calculate correlation coefficient if we have any data
        if diversity_nonan.shape[0] > 0:
            # calculate the correlation coefficient (returns an object)
            corr = pearsonr(diversity_nonan, accuracy_nonnan)

            if corr.pvalue < 0.05:
                pval = "*"
            elif corr.pvalue < 0.01:
                pval = "**"
            else:
                pval = ""

            print("& {0:.2f}{1} ".format(abs(corr.statistic), pval), end="")
        else:
            print("& ", end="")

    def printResults(self, output="test_acc", image_size=28):
        """
        Print a table of results in latex format and save to a text file if specified
        :return:
        """
        assert output in ["test_acc", "test_auc", "generalisation_gap"], \
            "Please set the plotting metric to either 'test_accuracy' or 'valid_accuracy' or 'generalisation_gap'"

        # Get the names of the datasets present in results. We will have a separate column for each dataset
        dataset_names = np.unique(self.results["dataset_name"].values)

        # print the first few lines of the latex table
        print(r"\begin{tabular}{p{3.4cm}|p{0.7cm}p{0.7cm}p{0.7cm}|p{0.7cm}|p{0.7cm}p{0.7cm}p{0.7cm}|p{0.7cm}|p{0.7cm}p{0.7cm}p{0.7cm}|p{0.7cm}|}")
        print(r" &  \multicolumn{4}{|c|}{MNIST} & \multicolumn{4}{|c|}{EMNIST} &\multicolumn{4}{|c|}{PneuMNIST}\\")
        print(r"\hline")
        print(r"No. Samples & 500 & 1000 & 2000 & all & 500 & 1000 & 2000 & all & 200 & 500 & 1000 & all \\")
        print(r"\hline")

        # iterate over the diversity scoring metrics
        for score, score_name in zip(self.diversity_scores, self.plot_titles):
            print(score_name, end="")
            # iterate over the datasets
            for dataset_name in dataset_names:
                # find the range of dataset sizes used for this dataset
                n_samples = np.unique(self.results["n_samples"][self.results["dataset_name"] == dataset_name].values)

                # iterate over the number of samples
                for ns in n_samples:
                    # filter the results by dataset, diversity metric and number of samples
                    condition_1 = self.results["dataset_name"] == dataset_name
                    condition_2 = self.results["n_samples"] == ns
                    condition_3 = self.results["image_size"] == image_size
                    diversity = self.results[score][condition_1 & condition_2 & condition_3]
                    accuracy = self.results[output][condition_1 & condition_2 & condition_3]

                    self.__printCorrelation__(diversity, accuracy)

                # Get a correlation value for all samples
                condition_1 = self.results["dataset_name"] == dataset_name
                diversity = self.results[score][condition_1]
                accuracy = self.results[output][condition_1]

                self.__printCorrelation__(diversity, accuracy)

                print("", end="")
            print("\\\\")

        # print the last part of the table in latex
        print(r"\end{tabular}")

def main():
    plotter = ResultsProcesser(experiment_name="GeneralisationDiversity")
    plotter.plot(output="test_AUC", dataset=["pneumoniamnist"], image_size=28)

    plotter.printResults(output="test_AUC", image_size=28)

    #plotter = ResultsProcesser(experiment_name="Generalisation_Fixed_Entropy")
    #plotter.plot(output="test_accuracy", dataset=["MNIST", "EMNIST"])


if __name__ == "__main__":
    main()