import subprocess
import argparse
import os
import random
import pickle as pkl

# Set up the argument parser
parser = argparse.ArgumentParser(description="Run experiments to determine the relationship between dataset diversity and generalisation performance")
parser.add_argument("-r", "--root_dir", type=str, help="Root directory where the code and data are located", default="/Users/katecevora/Documents/PhD")
parser.add_argument("-n", "--num_samples", type=int, help="Number of dataset samples per category to use", default=200)
parser.add_argument("-d", "--dataset", type=str, help="Dataset which we will use to run experiments", default="pneumoniamnist")
parser.add_argument("-i", "--image_size", type=int, help="Image size", default=128)

args = parser.parse_args()

code_dir = os.path.join(args.root_dir, "code/DatasetDiversityMedMNIST")
data_dir = os.path.join(args.root_dir, "data")
params_folder = os.path.join(code_dir, "params")

dataset = args.dataset
n_samples = args.num_samples
image_size = args.image_size


def main():
    seeds = [112, 234, 23, 453, 21, 12, 6, 2, 67, 88]

    if not(os.path.exists(params_folder)):
        os.mkdir(params_folder)

    for s in seeds:
        for diversity in ["high", "low", "random"]:

            # generate a unique ID for the classifier model
            unique_id = ''.join(random.choices('0123456789', k=6))

            params = {
                "n_samples": n_samples,
                "random_seed": s,
                "n_layers": 3,
                "n_epochs": 100,
                "n_workers": 0,
                "batch_size": 100,
                "model_name": "classifier_{}.pt".format(unique_id),
                "dataset_name": dataset,
                "diversity": diversity,
                "image_size": image_size
            }

            params_name = "{}_{}_{}_{}_{}.pkl".format(dataset, image_size, n_samples, s, diversity)

            f = open(os.path.join(params_folder, params_name), "wb")
            pkl.dump(params, f)
            f.close()

            print(
                "Running experiment with configuration: image size={0}, n_samples={1}, seed={2}, dataset={3}".format(
                    image_size, n_samples, s, dataset))

            command = ["python", "testGeneralisation.py", "-p", params_name, "-r", args.root_dir]

            # Run the command
            result = subprocess.run(command, capture_output=True, text=True)
            print(result.stdout)
            print(result.stderr)



if __name__ == "__main__":
    main()
