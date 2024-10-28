# class for constructing a 2D autoencoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    """
    Simple convolutional autoencoder model.
    """
    def __init__(self, save_path=""):
        super(ConvAutoencoder, self).__init__()
        # set the path to save the model
        self.save_path = save_path

        # Define the layers for the encoder
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        # Define the layers for the decoder
        # a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        """
        Forward function for the network.
        :param x: input to the network
        :return:
        x: output of the network
        x_compressed: reperesentation of the input from the mid layer.
        """
        # Encode
        # add hidden layers with relu activation function and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x_compressed = self.pool(x)  # compressed representation, we will extract this

        # Decode
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x_compressed))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))

        return x, x_compressed
