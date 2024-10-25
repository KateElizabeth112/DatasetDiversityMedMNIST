# Copied from the Vendi score library so we can modify locally
import numpy as np
import scipy
import scipy.linalg
from sklearn import preprocessing
from torchvision import transforms
from torchvision.models import inception_v3
import torch.nn as nn
import torch
from vendi_score import data_utils

def weight_K(K, p=None):
    if p is None:
        return K / K.shape[0]
    else:
        return K * np.outer(np.sqrt(p), np.sqrt(p))


def normalize_K(K):
    d = np.sqrt(np.diagonal(K))
    return K / np.outer(d, d)


def entropy_q(p, q=1):
    p_ = p[p > 0]
    if q == 1:
        return -(p_ * np.log(p_)).sum()
    if q == "inf":
        return -np.log(np.max(p))
    return np.log((p_ ** q).sum()) / (1 - q)


def score_K(K, q=1, p=None, normalize=False):
    if normalize:
        K = normalize_K(K)
    K_ = weight_K(K, p)
    if type(K_) == scipy.sparse.csr.csr_matrix:
        w, _ = scipy.sparse.linalg.eigsh(K_)
    else:
        w = scipy.linalg.eigvalsh(K_)
    return np.exp(entropy_q(w, q=q))


def score_X(X, q=1, p=None, normalize=True):
    if normalize:
        X = preprocessing.normalize(X, axis=1)
    K = X @ X.T
    return score_K(K, q=1, p=p)


def score_dual(X, q=1, normalize=True):
    if normalize:
        X = preprocessing.normalize(X, axis=1)
    n = X.shape[0]
    S = X.T @ X
    w = scipy.linalg.eigvalsh(S / n)
    m = w > 0
    return np.exp(entropy_q(w, q=q))


def score(samples, k, q=1, p=None, normalize=False):
    n = len(samples)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i, j] = K[j, i] = k(samples[i], samples[j])
    return score_K(K, p=p, q=q, normalize=normalize)


def intdiv_K(K, q=1, p=None):
    K_ = K ** q
    if p is None:
        p = np.ones(K.shape[0]) / K.shape[0]
    return 1 - np.sum(K_ * np.outer(p, p))


def intdiv_X(X, q=1, p=None, normalize=True):
    if normalize:
        X = preprocessing.normalize(X, axis=1)
    K = X @ X.T
    return intdiv(K, q=q, p=p)


def intdiv(samples, k, q=1, p=None):
    n = len(samples)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i, j] = K[j, i] = k(samples[i], samples[j])
    return intdiv_K(K, q=q, p=p)


def get_inception(pretrained=True, pool=True):

    model = inception_v3(pretrained=True, transform_input=True).eval()

    if pool:
        model.fc = nn.Identity()
    return model


def get_embeddings(
    images,
    model=None,
    transform=None,
    batch_size=64,
    device=torch.device("cpu"),
):
    if type(device) == str:
        device = torch.device(device)
    if model is None:
        model = get_inception(pretrained=True, pool=True).to(device)
        transform = inception_transforms()
    if transform is None:
        transform = transforms.ToTensor()
    embeddings = []
    for batch in data_utils.to_batches(images, batch_size):
        x = torch.stack([transform(img) for img in batch], 0).to(device)
        with torch.no_grad():
            output = model(x)
        if type(output) == list:
            output = output[0]
        embeddings.append(output.squeeze().cpu().numpy())
    return np.concatenate(embeddings, 0)


def inception_transforms():
    return transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, -1, -1)),
        ]
    )


def get_inception_embeddings(images, batch_size=64, device="cpu"):
    if type(device) == str:
        device = torch.device(device)
    model = get_inception(pretrained=True, pool=True).to(device)
    transform = inception_transforms()
    return get_embeddings(
        images,
        batch_size=batch_size,
        device=device,
        model=model,
        transform=transform,
    )


def embedding_vendi_score(
    images, batch_size=64, device="cpu", model=None, transform=None
):
    X = get_embeddings(
        images,
        batch_size=batch_size,
        device=device,
        model=model,
        transform=transform,
    )
    n, d = X.shape
    if n < d:
        return score_X(X)
    return score_dual(X)


def getInceptionEmbeddings( images, batch_size=64, device="cpu", model=None, transform=None):

    X = get_embeddings(
        images,
        batch_size=batch_size,
        device=device,
        model=model,
        transform=transform,
    )

    return X