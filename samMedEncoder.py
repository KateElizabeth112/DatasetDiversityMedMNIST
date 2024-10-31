import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
from SAM_Med2D.segment_anything.build_sam import sam_model_registry
from SAM_Med2D.segment_anything.predictor_sammed import SammedPredictor
from argparse import Namespace


class SamMedEncoder:
    def __init__(self, data, params):
        # check that the vectors parameter is a numpy array with two dimensions
        assert isinstance(params, dict), "params should be a dictionary"
        assert isinstance(data, Dataset), "train_data is not an instance of Dataset"

        self.params = params
        self.data = data

        # set up a data loader
        self.data_loader = torch.utils.data.DataLoader(self.data, batch_size=1)

    def encode(self):
        args = Namespace()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = "SAM_Med2D/pretrain_model/sam-med2d_b.pth"
        model = sam_model_registry["vit_b"](args).to(device)
        predictor = SammedPredictor(model)

        for k, (image, _) in enumerate(self.data_loader):
            image_np = np.moveaxis(image.squeeze(dim=0).numpy(), 0, 2)
            predictor.set_image(image_np)
            embedding = predictor.get_image_embedding()
            if k == 0:
                vectors = embedding.flatten().unsqueeze(0)
            else:
                vectors = torch.cat((vectors, embedding.flatten().unsqueeze(0)), dim=0)

        return vectors.detach().numpy()
