import copy

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
from SAM_Med2D.segment_anything.build_sam import sam_model_registry
from SAM_Med2D.segment_anything.predictor_sammed import SammedPredictor
from argparse import Namespace

import pickle as pkl
import os


class SamMedEncoder:
    def __init__(self, data, params, start_idx):
        # check that the vectors parameter is a numpy array with two dimensions
        assert isinstance(params, dict), "params should be a dictionary"
        assert isinstance(data, Dataset), "train_data is not an instance of Dataset"

        self.data = data
        self.code_dir = params["code_dir"]
        self.dataset_name = params["dataset_name"]
        self.image_size = params["image_size"]
        self.representations_dir = os.path.join(self.code_dir, "representations", "{}_{}".format(self.dataset_name, self.image_size))
        self.start_idx = start_idx

        if not os.path.exists(self.representations_dir):
            os.mkdir(self.representations_dir)

        # set up a data loader. batch size must be 1 for SAMMed Encoder
        self.data_loader = torch.utils.data.DataLoader(self.data, batch_size=1, shuffle=False)

    def encode(self):
        args = Namespace()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = os.path.join(self.code_dir, "SAM_Med2D/pretrain_model/sam-med2d_b.pth")
        model = sam_model_registry["vit_b"](args).to(device)
        predictor = SammedPredictor(model)

        for k, (image, _) in enumerate(self.data_loader):
            if k >= self.start_idx:
                print("Encoding {0} {1} image {2}".format(self.dataset_name, self.image_size, k))
                image_np = np.moveaxis(image.squeeze(dim=0).numpy(), 0, 2)
                predictor.set_image(image_np)
                embedding = predictor.get_image_embedding()

                f = open(os.path.join(self.representations_dir, "img_{}.pkl".format(k)), "wb")
                pkl.dump(embedding.flatten().unsqueeze(0), f)
                f.close()

    def retrieve(self, indices):
        # retrieve pre-computed embeddings based on a list of indicies
        for p in range(indices.shape[0]):
            f = open(os.path.join(self.representations_dir, "img_{}.pkl".format(indices[p])), "rb")
            embedding = pkl.load(f)
            f.close()

            if p == 0:
                vectors = copy.deepcopy(embedding)
            else:
                vectors = torch.cat((vectors, embedding), dim=0)

        return vectors.detach().numpy()
