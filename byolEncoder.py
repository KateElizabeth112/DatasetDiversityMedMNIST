import torch
from byol_pytorch import BYOL, BYOLTrainer
from torchvision import models


class BYOLEncoder:
    def __init__(self, data, params):
        self.params = params
        self.data = data

        # set up a data loader
        self.data_loader = torch.utils.data.DataLoader(self.data, batch_size=20)

    def encode(self):
        resnet = models.resnet50(pretrained=True)

        learner = BYOL(
            resnet,
            image_size=self.params["image_size"],
            hidden_layer='avgpool'
        )

        learner.load_state_dict(torch.load("./BYOL/models/byol_{0}_{1}_weights.pth".format(self.params["dataset_name"], self.params["image_size"])))

        # iterate over the batches in the dataset and store the embeddings in a stack
        for k, (images, _) in enumerate(self.data_loader):
            _, embeddings = learner(images, return_embedding=True)

            if k == 0:
                vectors = embeddings
            else:
                vectors = torch.cat((vectors, embeddings), dim=0)

        return vectors.detach().numpy()
