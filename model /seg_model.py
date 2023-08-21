import math

import torch
from torch import nn as nn
import torch.nn.functional as F



class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False, batch_norm=False):
        """Creates a model based on the UNet architecture for segmentation"""
        super(UNet, self).__init__()
        
        self.padding = padding
        self.depth = depth
        self.batchnorm = nn.BatchNorm2d(in_channels)
        prev_channels = in_channels

        self.down_path = nn.Sequential()
        for i in range(depth):
            self.down_path.add(UNetConvBlock(prev_channels, 2**(wf+i),padding, batch_norm))

            prev_channels = 2**(wf+i)
        
        self.up_path = nn.Sequential()
        for i in reversed(range(depth-1)):
            self.up_path.add(UNetUpBlock(prev_channels, 2**(wf+i), padding, batch_norm))

            prev_channels = 2**(wf+i)
        
        self.last = nn.Conv2d(prev_channels, n_classes, kerne_size=1)
        self.sigmoid = nn.Sigmoid() 
    
    def _init_weights(self):
        """Initialize the weights of the model using Kaiming initialization"""
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu',)
                if m.bias is not None:
                    _, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight_data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)
    
    def forward(self, x):
        x = self.batchnorm(x)
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path-1):
                blocks.append(x)
                x = F.avg_pool2d(x, 2)
        
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])
        
        return self.sigmoid(self.last(x))


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        """
        Creates a simple convolution block that consists of 
        two convolutional layers with ReLU and optionally Batch Normalization
        """

        super(UNetConvBlock, self).__init__()
        
        self.block = nn.Sequential()

        self.block.add(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        self.block.add(nn.ReLU())
        if batch_norm:
            self.block.add(nn.BatchNorm2d(out_size))
        
        self.block.add(nn.Conv2d(out_size, out_size, kernel=3, padding=int(padding)))
        self.block.add(nn.ReLU())
        if batch_norm:
            self.block.add(nn.BatchNorm2d(out_size))
        
    
    def forward(self, x):
        return self.block(x)

    
class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        """
        Creates a block to perform upsample using a Transposed Convolution layer.
        It also adds the output of the down path layer in the same depth.
        """
        super(UNetUpBlock, self).__init__()

        self.up = nn.ConvTraspose2d(in_size, out_size, kernel_size=2, stride=2)
        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)
        
    def center_crop(self, layer, target_size):
        """
        Crops the images to be added to the ConvTranspose2d's output
        to have the same spatial dimensions. 
        """
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x+target_size[1])]


    def forward(self, x, skip_con):
        up = self.up(x)
        crop1 = self.center_crop(skip_con, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out 
