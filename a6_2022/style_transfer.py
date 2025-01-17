"""
Implements a style transfer in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
import torch.nn as nn
from a6_helper import *

def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from style_transfer.py!')

def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_original: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    ############################################################################
    # TODO: Compute the content loss for style transfer.                       #
    ############################################################################
    loss = content_weight * torch.pow((content_current - content_original), 2).sum()
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return loss

def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    gram = None
    ############################################################################
    # TODO: Compute the Gram matrix from features.                             #
    # Don't forget to implement for both normalized and non-normalized version #
    ############################################################################
    N, C, H, W = features.shape
    features = features.view(N, C, -1)

    gram_matrix = torch.bmm(features, features.transpose(1,2))

    if normalize:
       gram_matrix /= H * W * C
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return gram_matrix


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the style loss at a set of layers.                        #
    # Hint: you can do this with one for loop over the style layers, and       #
    # should not be very much code (~5 lines).                                 #
    # You will need to use your gram_matrix function.                          #
    ############################################################################
    loss = 0.0

    for idx, li in enumerate(style_layers):
      content_gramm = gram_matrix(feats[li])
      style_gramm = style_targets[idx]
      weight = style_weights[idx]
      curr_loss = weight * torch.pow((content_gramm - style_gramm), 2).sum()
      loss += curr_loss

    return loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    ############################################################################
    # TODO: Compute total variation loss.                                      #
    # Your implementation should be vectorized and not require any loops!      #
    ############################################################################
    # Sum through the height.
    sumh = torch.sum((img[..., 1:, :] - img[..., :-1, :]) ** 2)
    # Sum through the width.
    sumw = torch.sum((img[..., 1:] - img[..., :-1]) ** 2)
    # Compute the loss.
    loss = tv_weight * (sumh + sumw)
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return loss

def guided_gram_matrix(features, masks, normalize=True):
  """
  Inputs:
    - features: PyTorch Tensor of shape (N, R, C, H, W) giving features for
      a batch of N images.
    - masks: PyTorch Tensor of shape (N, R, H, W)
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, R, C, C) giving the
      (optionally normalized) guided Gram matrices for the N input images.
  """
  guided_gram = None
  ##############################################################################
  # TODO: Compute the guided Gram matrix from features.                        #
  # Apply the regional guidance mask to its corresponding feature and          #
  # calculate the Gram Matrix. You are allowed to use one for-loop in          #
  # this problem.                                                              #
  ##############################################################################
  N, R, C, H, W = features.shape

  masks = masks.unsqueeze(2)
  guided_features = torch.mul(features, masks)
  guided_gram = torch.zeros(N, R, C, C)

  for feature_idx in range(R):
    subfeatures = guided_features[:, feature_idx, :, :, :].view(N, C, -1)
    gramm_matrix = torch.bmm(subfeatures, subfeatures.transpose(1,2))

    if normalize:
      gramm_matrix /= (H * W * C)

    guided_gram[:, feature_idx, :, :] = gramm_matrix
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return guided_gram


def guided_style_loss(feats, style_layers, style_targets, style_weights, content_masks):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the guided Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
    - content_masks: List of the same length as feats, giving a binary mask to the
      features of each layer.
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the guided style loss at a set of layers.                 #
    ############################################################################
    loss = 0.0
    for idx, layer_idx in enumerate(style_layers):
      features = feats[layer_idx]
      masks = content_masks[layer_idx]
      content_gramm = guided_gram_matrix(features, masks)

      target_gramm = style_targets[idx]
      weight = style_weights[idx]

      curr_loss = weight * torch.pow((target_gramm - content_gramm), 2).sum()
      loss += curr_loss

    return loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
