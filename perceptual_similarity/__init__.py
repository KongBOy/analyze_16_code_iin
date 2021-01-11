from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from .models import dist_model

class PerceptualLoss(torch.nn.Module):
    def __init__(self, model='net-lin', net='vgg', use_gpu=True,
                 device = None, resnet50_model_name = None, **kwargs):
    # VGG using our perceptually-learned weights (LPIPS metric)
    # def __init__(self, model='net', net='vgg', use_gpu=True): # "default" way of using VGG
        super().__init__()
        print('Setting up Perceptual loss...')
        self.model = dist_model.DistModel()
        self.model.initialize(model=model, net=net, use_gpu=use_gpu, device = device, resnet50_model_name = resnet50_model_name, **kwargs)
        print('...Done')

    def forward(self, pred, target, normalize=False):
        """
        Pred and target are Variables.
        If normalize is on, assumes the images are between [0,1] and then scales thembetween [-1, 1]
        If normalize is false, assumes the images are already between [-1,+1]

        Inputs pred and target are NxCxHxW
        Output pytorch Variable N long

        If C=1, replicate channel dimension to allow for general vgg/resnet/... usage in perceptual loss
        """
        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1

        if pred.size(1) == 1:
            assert target.size(1) == 1
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        dist = self.model.forward_pair(target, pred)

        return dist
