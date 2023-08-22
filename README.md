# LUNA_Dataset

The Lung Node Analysis dataset consists of 888 Computed Tomography (CT) scans in mhd format and the main goal is to idnetify possible locations of nodules. The model is based on a UNet architecture and is used for a segmentation of each CT scan. For each pixel, the segmentation process assigns a probability of being a part of a nodule. The model can be trained to achieve high recall and then other procedures can be used for false positive reduction.  
