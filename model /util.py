
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms



IrcTuple = collection.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collection.namedtuple('XyzTuple', ['x', 'y', 'z'])


def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction):
    cri = np.array(coord_irc)[::-1]
    origin = np.array(origin_xyz)
    vxSize = np.array(vxSize_xyz)
    coords_xyz = (direction @ (cri * vxSize)) + origin
    return XyzTuple(*coords_xyz)

def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction):
    origin = np.array(origin_xyz)
    vxSize = np.array(vxSize_xyz)
    coord = np.array(coord_xyz)
    cri = ((coord - origin) @ np.linalg.inv(direction)) / vxSize
    cri = np.round(cri)
    return IrcTuple(int(cri[2]), int(cri[1]), int(cri[0]))


def augmentation(inputs, labels, flip=None, offset=None, scale=None, rotation=None, noise=None):
    
    transformation = transforms.Compose([
        transforms.RandomHorizontalFlipping(flip),
        transforms.Resize(scale),
        transforms.RandomRotation(rotation),
    ])

    transformation.to(inputs.device)
    labels = labels.to(torch.float32)

    inputs = transformation(inputs)
    labels = transformation(labels)

    if noise:
        inputs += noise * torch.randn_like(inputs)
    
    return inputs, labels > 0.5