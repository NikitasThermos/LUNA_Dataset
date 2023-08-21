import os 
import glob
import functools
import csv
import random

from collection import namedtuple

import numpy as np
import SimpleTK as sitk

import torch 
import torch.cuda
from torch.utils.data import Dataset

from util import XyzTuple, xyz2irc


CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule, hasAnnotation, diameter_nm, series_uid, center_xyz'
)

@functools.lru_cache(1)
def getCandidateInfoList():
    """
    Returns a list of info for each nodule and candidate. 
    Reads from two files, one for nodules and one for candidates.
    Returns the sorted list to balance the training/validation split.
    """

    mhd_list = glob.glob('data/luna/subset*/*.mhd')
    presentedOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    candidateInfo_list = []

    with open('data/luna/annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            center_xyz = tuple([float(x) for x in row[1:4]])
            diameter_nm = float(row[4])

            candidateInfo_list.append(
                CandidateInfoTuple(
                    True,
                    True,
                    diameter_nm,
                    series_uid,
                    center_xyz,
                )
            )
    
    with open('data/luna/candidate.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentedOnDisk_set:
                continue

            isNodule = bool(int(row[4]))
            center_xyz = tuple([float(x) for x in row[1:4]])

            if not isNodule:
                candidateInfo_list.append(
                    CandidateInfoTuple(
                        False,
                        False,
                        0.0,
                        series_uid,
                        center_xyz,
                    )
                )

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

@functools.lru_cache(1)
def getCandidateInfoDict():
    candidateInfo_list = getCandidateInfoList()
    candidateInfo_dict = {}

    for tuple in candidateInfo_list:
        candidateInfo_dict.setdefault(tuple.series_uid, []).append(tuple)
    
    return candidateInfo_dict

class Ct: 
    def __init__(self, series_uid):
        """
        Based on the uid gets a Computed Tomography (CT).
        It uses SimpleITK to read the images.
        """

        mhd_path = glob.glob('data/luna/subest*/{}.mhd'.format(series_uid))[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a


        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection().reshape(3, 3))

        candidateInfo_list = getCandidateInfoDict()[self.series_uid]

        # Store the positive nodules
        self.positiveInfo_list = [
            candidate_tup
            for candidate_tup in candidateInfo_list
            if candidate_tup.isNodule
        ]

        #Get the masks for the positive nodules on the CT based on the positiveInfo_list
        self.positive_mask = self.builAnnotationMask(self.positiveInfo_list)
        self.positive_indexes = (self.positive_mask.sum(axis=(1, 2)).nonzero()[0].tolist())

    def getRawCandidate(self, center_xyz, width_irc):
        """
        Get a candidate nodule with a specific center on the CT scan.
        First transforms the XYZ coordinates to Index, Column, Row and then
        it returns this specific chunnk and the corresponding mask.
        """
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = []

        #for each axis gets the center value and calculates the indexes based on the width.
        #also checks if indexes are out of bounds.
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])
            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0 :
                start_ndx = 0 
                end_ndx = int(width_irc)
            
            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(end_ndx - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))


        ct_chunk = self.hu_a[tuple(slice_list)]
        pos_chunk = self.positive_mask[tuple(slice_list)]

        return ct_chunk, pos_chunk, center_irc
    
    def buildAnnotationMask(self, positiveInfo_list, threshold_hu = -700):
        """
        Build masks that annotate which pixels of the CT are part of a nodule.
        Starting from a center of a nodule it looks at every direction and it annotates
        as positive if the intensity of the pixel is bigger than the threshold.
        Masks are used as labels for segmentation training. 
        """
        boundingBox_a = np.zeros_like(self.hu_a, dtype=np.bool)

        for candidateInfo_tup in positiveInfo_list:
            center_irc = xyz2irc(
                candidateInfo_tup.center_xyz,
                self.origin_xyz,
                self.vxSize_xyz,
                self.direction_a,
            )

            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            index_radius = 2
            try:
                while self.hu_a[ci + index_radius, cr, cc] > threshold_hu and self.hu_a[ci - index_radius, cr, cc] > threshold_hu:
                    index_radius += 1
            except:
                index_radius -= 1
            
            row_radius = 2
            try:
                while self.hu_a[ci, cr + row_radius, cc] > threshold_hu and self.hu_a[ci, cr + row_radius, cc] > threshold_hu:
                    row_radius += 1
            except:
                row_radius -= 1

            col_radius = 2
            try:
                while self.hu_a[ci, cr, cc + col_radius] > threshold_hu and self.hu_a[ci, cr, cc + col_radius] > threshold_hu:
                    col_radius += 1
            except:
                col_radius -= 1
            
            boundingBox_a[ci - index_radius: ci + index_radius + 1,
                          cr - row_radius : cr + row_radius + 1,
                          cc - col_radius : cc + col_radius + 1] = True
            

            #Because we create a box around the candidate in the previous step
            #the corners of the box may include non-nodule pixels that we remove
            #by calculating the union of the box and all high intensity pixels.
            mask_a = boundingBox_a & (self.hua_a > threshold_hu)

            return mask_a


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, pos_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    ct_chunk.clip(-1000, 1000)
    return ct_chunk, pos_chunk, center_irc

def getCtSampleSize(series_uid):
    ct = Ct(series_uid)
    return int(ct.hu_a.shape(0)), ct.positive_indexes

class Luna2dSegmentationDataset(Dataset):
    def __init__(self, val_stride=0, isValSet=None, series_uid=None, contextSlices=3, fullCt=False):
        """
        Creates a segmentation dataset of Cts. 
        """
        self.contextSlices = contextSlices
        self.fullCt = fullCt

        #Select a specific CT for debugging
        if series_uid :
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(getCandidateInfoDict().keys())

        #If this is validation set keep every nth item, else delete every nth item to
        #create the training dataset
        if isValSet:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list
        
        self.sample_list = []
        for series_uid in self.series_list:
            index_count, positive_indexes = getCtSampleSize(series_uid)

            #Get every slice of a CT or only the slices that contain 
            #positive pixels
            if self.fullCt:
                self.sample_list += [(series_uid, slice_ndx)
                                   for slice_ndx in range(index_count)]
            else: 
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in positive_indexes]
        
        self.candidateInfo_list = getCandidateInfoList()

        series_set = set(self.series_list)
        self.candidateInfo_list = [cit for cit in self.candidateInfo_list
                                   if cit.series_uid in series_set]
        self.pos_list = [nt for nt in self.candidateInfo_list
                         if nt.isNodule]
    
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        series_uid, slice_ndx = self.sample_list[index % len(self.sample_list)]
        return self.getitem_fullSlice(series_uid, slice_ndx)
    
    def shuffleSamples(self):
        random.shuffle(self.candidateInfo_list)
        random.shuffle(self.pos_list)
    
    def getitem_fullSlice(self, series_uid, slice_ndx):
        """
        Gets the slice for the segmentation. For better results the model is given 
        the neighborhood slices of the slice it tries to segment. For example, if 
        contextSlices = 3 we will return the 3 slices above and below from the slice that 
        the model works on. The context slices are given in the 'channel' axis. 
        """
        ct = getCt(series_uid)
        ct_t = torch.zeros((self.contextSlices * 2 + 1, 512, 512))

        start_ndx = slice_ndx - self.contextSlices
        end_ndx = slice_ndx + self.contextSlices

        #check if the slice index is out of bounds. If it is we repeat the last slice.
        for i, context_ndx in enumerate(range(start_ndx, end_ndx + 1)):
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)
            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx]).astype(np.float32)
        ct_t.clamp_(-1000, 1000)

        #Get the segmentation labels only for the slice that we are working on. 
        pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0)

        return ct_t, pos_t, ct.series_uid, slice_ndx
    

class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
    def __init__(self, *args, **kwargs):
        """
        Creates a dataset that works better for training.
        In this case it only returns croped images around the nodules
        instead of full slices. 
        """ 
        super().__init__(*args, **kwargs)
        self.ratio_int = 2

    def __len__(self):
        return 30000
    
    def __getitem__(self, ndx):
        candidateInfo_tup = self.pos_list[ndx % len(self.pos_list)]
        return self.getitem_trainingCrop(candidateInfo_tup)
    
    def getitem_trainingCrop(self, candidateInfo_tup):
        ct_a, pos_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            (7, 96, 96)
        )

        #Get the mask only for the center slice
        pos_a = pos_a[3:4]

        #random offsets to crop the image 
        row_offset = random.randrange(0, 32)
        col_offset = random.randrange(0, 32)

        ct_t = torch.numpy(ct_a[:, row_offset:row_offset+64, col_offset:col_offset+64])
        pos_t = torch.numpy(pos_a[:, row_offset:row_offset+64, col_offset:col_offset+64])

        slice_ndx = center_irc.index

        return ct_t, pos_t, candidateInfo_tup.series_uid, slice_ndx