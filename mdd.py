import numpy as np
import torch
import os

def _gaussian_kernel_matrix(x, y, sigmas):
    """
    Computes the Gaussian Kernel matrix
    """
    dist = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=-1)[:, :, None]
    beta = 1.0 / (2.0 * torch.Tensor(sigmas)[None, :])
    s = torch.matmul(dist, beta)
    k = torch.exp(-s).sum(axis=-1)
    return k

def _mmd(embedding, factor_z=2):
    """
    Computes the Maximum-Mean-Discrepancy (MMD): MMD(embedding,z) where
    z follows a standard normal distribution
    Computation is performed for multiple scales
    """
    # z = torch.randn(embedding.shape)
    z = torch.randn(embedding.shape[0] * factor_z, embedding.shape[1])
    sigmas = [
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        100,
        1e3,
        1e4,
        1e5,
        1e6,
    ]

    loss = torch.mean(_gaussian_kernel_matrix(embedding, embedding, sigmas))
    loss += torch.mean(_gaussian_kernel_matrix(z, z, sigmas))
    loss -= 2 * torch.mean(_gaussian_kernel_matrix(embedding, z, sigmas))
    return loss

def _gaussian_kernel_matrix(x, y, sigmas):
    """
    Computes the Gaussian Kernel matrix
    """
    dist = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=-1)[:, :, None]
    beta = 1.0 / (2.0 * torch.Tensor(sigmas)[None, :])
    s = torch.matmul(dist, beta)
    k = torch.exp(-s).sum(axis=-1)
    return k

def _mmd(embedding, factor_z=2):
    """
    Computes the Maximum-Mean-Discrepancy (MMD): MMD(embedding,z) where
    z follows a standard normal distribution
    Computation is performed for multiple scales
    """
    # z = torch.randn(embedding.shape)
    z = torch.randn(embedding.shape[0] * factor_z, embedding.shape[1])
    sigmas = [
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        100,
        1e3,
        1e4,
        1e5,
        1e6,
    ]

    loss = torch.mean(_gaussian_kernel_matrix(embedding, embedding, sigmas))
    loss += torch.mean(_gaussian_kernel_matrix(z, z, sigmas))
    loss -= 2 * torch.mean(_gaussian_kernel_matrix(embedding, z, sigmas))
    return loss

# Here you have to include your own path
# dataset_path = "/home/DATA/ITWM/lorenzp/"
dataset_path = "/home/lorenzp/workspace/competence_estimation/features/cifar10"
model = "resnet18"
mode = "benign"

# Features/logits/labels Trainings datad
features_id_train =  torch.load(f"{dataset_path}/{mode}/features_{mode}_{model}_train.pt")[:4000]
features_id_test  =  torch.load(f"{dataset_path}/{mode}/features_{mode}_{model}_test.pt")[:4000]

print("train: ", _mmd(torch.from_numpy(features_id_train), factor_z=2))
print("test:  ", _mmd(torch.from_numpy(features_id_test), factor_z=2))