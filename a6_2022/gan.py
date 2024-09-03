from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim

NOISE_DIM = 96


def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device="cpu"):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - noise_dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, noise_dim) containing uniform
      random noise in the range (-1, 1).
    """
    noise = None
    ##############################################################################
    # TODO: Implement sample_noise.                                              #
    ##############################################################################
    noise = 2 * torch.rand((batch_size, noise_dim), dtype=dtype, device=device) - 1
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################

    return noise


def discriminator():
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement discriminator.                                           #
    ############################################################################
    layers = []
    layers.append(nn.Linear(784, 256))
    layers.append(nn.LeakyReLU(negative_slope=0.01))
    layers.append(nn.Linear(256, 256))
    layers.append(nn.LeakyReLU())
    layers.append(nn.Linear(256, 1))

    model = nn.Sequential(*layers)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return model


def generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement generator.                                               #
    ############################################################################
    layers = []
    layers.append(nn.Linear(noise_dim, 1024))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(1024, 1024))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(1024, 784))
    layers.append(nn.Tanh())

    model = nn.Sequential(*layers)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement discriminator_loss.                                        #
    ##############################################################################
    real_labels = torch.ones_like(logits_real)
    fake_labels = torch.zeros_like(logits_fake)

    loss_real = torch.nn.functional.binary_cross_entropy_with_logits(logits_real, real_labels)
    loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(logits_fake, fake_labels)

    loss = loss_real + loss_fake
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement generator_loss.                                            #
    ##############################################################################
    ones_fake = torch.ones_like(logits_fake)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_fake, ones_fake)
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = None
    ##############################################################################
    # TODO: Implement optimizer.                                                 #
    ##############################################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5,0.99))
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return optimizer


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    ##############################################################################
    # TODO: Implement ls_discriminator_loss.                                     #
    ##############################################################################
    loss_real = 1 / 2 * torch.pow((scores_real - 1), 2).mean()
    loss_fake = 1 / 2 * torch.pow(scores_fake, 2).mean()

    loss = loss_real + loss_fake    
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    ##############################################################################
    # TODO: Implement ls_generator_loss.                                         #
    ##############################################################################
    loss = 1 / 2 * torch.pow((scores_fake - 1), 2).mean()
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def build_dc_classifier():
    """
    Build and return a PyTorch nn.Sequential model for the DCGAN discriminator
    implementing the architecture in the notebook.
    """

    model = None
    ############################################################################
    # TODO: Implement build_dc_classifier.                                     #
    ############################################################################
    layers = []
    layers.append(nn.Unflatten(dim=1, unflattened_size=(1,28,28)))
    layers.append(nn.Conv2d(1, 32, kernel_size=5, stride=1))
    layers.append(nn.LeakyReLU(negative_slope=0.01))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    layers.append(nn.Conv2d(32, 64, kernel_size=5, stride=1))
    layers.append(nn.LeakyReLU(negative_slope=0.01))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    layers.append(nn.Flatten(start_dim=1))
    layers.append(nn.Linear(1024, 4*4*64))
    layers.append(nn.LeakyReLU(negative_slope=0.01))
    layers.append(nn.Linear(4*4*64, 1))

    model = nn.Sequential(*layers)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def build_dc_generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the DCGAN
    generator using the architecture described in the notebook.
    """

    # * Fully connected with output size 1024
    # * `ReLU`
    # * BatchNorm
    # * Fully connected with output size 7 x 7 x 128 
    # * `ReLU`
    # * BatchNorm
    # * Reshape into Image Tensor of shape 7 x 7 x 128
    # * Conv2D^T (Transpose): 64 filters of 4x4, stride 2, 'same' padding (use `padding=1`)
    # * `ReLU`
    # * BatchNorm
    # * Conv2D^T (Transpose): 1 filter of 4x4, stride 2, 'same' padding (use `padding=1`)
    # * `TanH`
    # * Should have a 28 x 28 x 1 image, reshape back into 784 vector
    model = None
    ############################################################################
    # TODO: Implement build_dc_generator.                                      #
    ############################################################################
    layers = []
    layers.append(nn.Linear(noise_dim, 1024))
    layers.append(nn.ReLU())
    layers.append(nn.BatchNorm1d(1024))
    layers.append(nn.Linear(1024, 7*7*128))
    layers.append(nn.ReLU())
    layers.append(nn.BatchNorm1d(7*7*128))
    layers.append(nn.Unflatten(dim=1, unflattened_size=(128,7,7)))
    layers.append(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1))
    layers.append(nn.ReLU())
    layers.append(nn.BatchNorm2d(64))
    layers.append(nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1))
    layers.append(nn.Tanh())
    layers.append(nn.Flatten())

    model = nn.Sequential(*layers)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model
