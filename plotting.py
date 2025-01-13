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

        # Add label entropy and domain gap
        plot_titles.append("Label Entropy")
        diversity_scores.append("label_entropy_train")
        plot_titles.append("Domain Gap")
        diversity_scores.append("domain_gap")

        self.diversity_scores = diversity_scores
        self.plot_titles = plot_titles

    def plot(self, output="test_acc", dataset="", image_size=28, ns=200):
        assert isinstance(dataset, list), "Please specify the dataset/s within a list"

        # Check that the specified dataset/s are present in the results file
        for ds in dataset:
            assert ds in list(np.unique(self.results["dataset_name"].values)), "Dataset {} not found in results".format(ds)

        fig, axes = plt.subplots(nrows=4, ncols=3, sharey=True)

        # list of colours to use for plotting different datasets
        colours_list = [lred, lblu, lgrn]

        # check how many different sample sizes we have for this dataset and image size combo
        condition1 = self.results["dataset_name"] == ds
        condition2 = self.results["image_size"] == image_size
        n_samples = np.unique(self.results["n_samples"][condition1 & condition2].values)
        #diversity_types = np.unique(self.results["diversity"][condition1 & condition2].values)
        diversity_types = ["high", "random", "low"]
        diversity_colours = ["r", "g", "b"]

        for i in range(len(self.diversity_scores)):
            # Check that we have this column present in the results CSV, if not, just skip
            if self.diversity_scores[i] in self.results.columns:
                ax = axes.flat[i]

                # iterate over the number of samples in the training dataset

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
        #ax = axes.flat[i+1]
        #ax.scatter([], [], color=dc, label="n_samples={0}".format(ns))
        #ax.axis("off")
        #ax.legend()

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

    def printCorrelation(self, output="test_AUC", dataset_name="pneumoniamnist", image_sizes=[28]):
        """
        Print a table of results in latex format and save to a text file if specified
        :return:
        """
        assert output in ["test_acc", "test_AUC", "gap"], \
            "Please set the plotting metric to either 'test_accuracy' or 'valid_accuracy'"

        diversity_metrics = ["vs", "intdiv"]
        encoders = ["pixel", "auto", "inception", "random", "sammed"]
        additional_metrics = ["label_entropy_train", "domain_gap"]

        # print the first few lines of the latex table
        n_samples_per_image_size = []
        total_experiments_counter = 0
        print(r"\begin{tabular}{|p{2.5cm}|p{2cm}|", end="")
        for image_size in image_sizes:
            condition1 = self.results["dataset_name"] == dataset_name
            condition2 = self.results["image_size"] == image_size
            n_samples = np.unique(self.results["n_samples"][condition1 & condition2].values)
            total_experiments_counter += n_samples.shape[0]
            n_samples_per_image_size.append(n_samples)
            for ns in n_samples:
                print(r"p{1.2cm}", end="")
            print("|", end="")
        print(r"}")

        print(r"\hline")

        print(r" &  ", end="")
        for i, image_size in enumerate(image_sizes):
            n_samples = n_samples_per_image_size[i]
            print(r"& \multicolumn{" + str(n_samples.shape[0]) + r"}{|c|}{Image Size = " + str(image_size) + "}", end="")
        print(r"\\")

        print(r"Metric & Encoder ", end="")
        for i, image_size in enumerate(image_sizes):
            for n in n_samples_per_image_size[i]:
                print(r" & $N_s$=" + str(n), end="")
        print(r" \\")


        print(r"\hline")

        # iterate over the diversity scoring metrics
        #for score, score_name in zip(self.diversity_scores, self.plot_titles):
        for metric in diversity_metrics:
            for encoder in encoders:
                print("{0} & {1} ".format(metric, encoder), end="")
                score = "{0}_{1}_train".format(metric, encoder)
                for image_size in image_sizes:
                    # find the range of dataset sizes used for this dataset
                    condition1 = self.results["dataset_name"] == dataset_name
                    condition2 = self.results["image_size"] == image_size
                    n_samples = np.unique(self.results["n_samples"][condition1 & condition2].values)

                    # cycle over the number of samples in the training dataset
                    for ns in n_samples:
                        # get the scores for the diversity metric
                        condition1 = self.results["dataset_name"] == dataset_name
                        condition2 = self.results["image_size"] == image_size
                        condition3 = self.results["n_samples"] == ns

                        diversity = self.results[score][condition1 & condition2 & condition3]
                        if output == "gap":
                            accuracy = self.results["val_acc"][condition1 & condition2 & condition3] - \
                                       self.results["test_acc"][condition1 & condition2 & condition3]
                        else:
                            accuracy = self.results[output][condition1 & condition2 & condition3]

                        self.__printCorrelation__(diversity, accuracy)

                print("\\\\")
            print("\hline")

        for score in additional_metrics:
            print("{0} & n/a ".format(score), end="")

            for image_size in image_sizes:
                # find the range of dataset sizes used for this dataset
                condition1 = self.results["dataset_name"] == dataset_name
                condition2 = self.results["image_size"] == image_size
                n_samples = np.unique(self.results["n_samples"][condition1 & condition2].values)

                # cycle over the number of samples in the training dataset
                for ns in n_samples:
                    # get the scores for the diversity metric
                    condition1 = self.results["dataset_name"] == dataset_name
                    condition2 = self.results["image_size"] == image_size
                    condition3 = self.results["n_samples"] == ns

                    diversity = self.results[score][condition1 & condition2 & condition3]
                    if output == "gap":
                        accuracy = self.results["val_acc"][condition1 & condition2 & condition3] - \
                                   self.results["test_acc"][condition1 & condition2 & condition3]
                    else:
                        accuracy = self.results[output][condition1 & condition2 & condition3]

                    try:
                        self.__printCorrelation__(diversity, accuracy)
                    except:
                        print(" & ", end="")

            print("\\\\")

        print(r"\hline")
        print(r"\end{tabular}")

    def printPeformance(self, output="test_AUC", dataset_names=["pneumoniamnist", "chestmnist"], image_sizes=[28, 128], ns=[50, 200, 500]):
        # print latex for the first few lines of the table
        print(r"\begin{tabular}{|p{2.5cm}|", end="")
        for image_size in image_sizes:
            for n in ns:
                print(r"p{1.7cm}|", end="")
        print(r"}")

        print(r"\hline")

        for image_size in image_sizes:
            print(r" & \multicolumn{" + str(len(ns)) + r"}{|c|}{Image Size = " + str(image_size) + "}", end="")
        print(r" \\")

        print(r"\hline")

        print("Dataset", end="")
        for image_size in image_sizes:
            for n in ns:
                print(r" & $N_s$=" + str(n), end="")
        print(r" \\")

        print(r"\hline")

        for dataset_name in dataset_names:
            print("{}".format(dataset_name), end="")
            for image_size in image_sizes:
                for n in ns:
                    condition1 = self.results["dataset_name"] == dataset_name
                    condition2 = self.results["image_size"] == image_size
                    condition3 = self.results["n_samples"] == n

                    result = self.results[output][condition1 & condition2 & condition3].values

                    print(" & {0:.2f} ({1:.2f})".format(np.nanmean(result), np.nanstd(result)), end="")
            print(r" \\")
        print(r"\hline")
        print(r"\end{tabular}")


def main():
    plotter = ResultsProcesser(experiment_name="GeneralisationDiversity")

    #plotter.printCorrelation( output="test_acc", dataset_name="chestmnist", image_sizes=[28, 128])

    #plotter.printPeformance(output="test_AUC", dataset_names=["pneumoniamnist", "chestmnist"], image_sizes=[28, 128], ns=[50, 200, 500])

    plotter.plot(output="val_AUC", dataset=["breastmnist"], image_size=28, ns=200)


if __name__ == "__main__":
    main()