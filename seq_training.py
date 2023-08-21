import argparse
import datetime
import hashlib
import os
import shutil
import socket
import sys


from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from dsets import Luna2dSegmentationDataset, TrainingLuna2dSegmentationDataset, getCt
from utils.py import augmentation
from model.py import UNet



METRICS_LOSS_NDX = 0
METRICS_TP_NDX = 1
METRICS_FN_NDX = 2
METRICS_FP_NDX = 3

METRICS_SIZE = 4

class SegTraining:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        
        parser = argparse.ArgumentParser()

        parser.add_argument('--batch-size',
            help='Training batch size',
            default=16,
            type=int,
        )

        parser.add_argument('--num-workers',
            help='Data loading worker processes',
            default=4,
            type=int,
        )

        parser.add_argument('--epochs',
            help='Training epochs',
            default=1,
            type=int,
        )

        parser.add_argument('--flip',
            help="Flip the training data .",
            action='store_true',
            default=True,
        )


        parser.add_argument('--scale',
            help="Scale the training data.",
            default=0.2,
            type=float,  
        )

        parser.add_argument('--rotate',
            help="Rotate the training data.",
            action='store_true',
            default=True,
        )

        parser.add_argument('--noise',
            help="Add noise to the training data .",
            default=25.0,
            type=float,
        )

        
        parser.add_argument('--tb-prefix',
            default='luna',
            help="Prefix for Tensorboard.",
        )

        parser.add_argument('--comment',
            help="Comment suffix for Tensorboard.",
            nargs='?',
            default='none',
        )
        
        
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.totalTrainingSamples_count = 0 
        self.trn_writer = None
        self.val_writer = None
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")


        self.segmentation_model = UNet(in_channel=7, n_classes=1, wf=4, padding=True, batch_norm=True, up_mode='conv')
        self.segmentation_model.to(self.device)
    

        self.optimizer = Adam(self.segmentation_model.parameters())


    def initDataloader(self,isValSet, val_stride=10, context_slices=3):
        if isValSet:
            dataset = Luna2dSegmentationDataset(
                val_stride=val_stride,
                isValSet=isValSet,
                contextSlices=context_slices
            )
        else:
            dataset = TrainingLuna2dSegmentationDataset(
                val_stride=val_stride,
                isValSet=isValSet,
                contextSlices=context_slices
            )

        return DataLoader(
            dataset,
            batch_size = self.cli_args.batch_size,
            num_workers = self.cli_args.num_workers,
            pin_memory = self.use_cuda,
        )
    
    def initTensorboardWriter(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir = log_dir + '_train_' + self.cli_args.comment
            )
            self.val_writer = SummaryWriter(
                log_dir = log_dir + '_val_' + self.cli_args.comment
            )


    def diceLoss(self, prediction, label, epsilon=1):
        """Calculates the dice loss between the prediction of the model and the ground truth"""
        diceLabel = label.sum(dim=[1, 2, 3])
        dicePrediction = prediction.sum(dim=[1, 2, 3])
        diceCorrect = (prediction * label).sum(dim=[1, 2, 3])

        diceRatio = (2 * diceCorrect + epsilon) / (dicePrediction + diceLabel + epsilon)

        return 1 - diceRatio
        
    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics, classificationThreshold=0.5):
        """
        Computes the loss of a batch. Calling the dice loss twice.
        First for the normal loss between the predictions and the labels and then
        only for the pixels that are positive on the label mask. The second one will 
        generate loss only for the false negatives. 
        """
        input, label, _, _ = batch_tup

        input = input.to(self.device)
        label = label.to(self.device)
        
        
        if self.segmentation_model.training:
            input, label = augmentation(input, label)

        prediction = self.segmentation_model(input)

        diceLoss = self.diceLoss(prediction, label) #normal Dice loss 
        fnLoss = self.diceLoss(prediction * label, label)#Dice loss only for the positive pixels

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input.size(0)

        #Keep metrics based on the predictions of the model
        with torch.no_grad():
            predictionBool = (prediction[:, 0:1] > classificationThreshold.to(torch.float32))

            tp = (predictionBool * label).sum(dim=[1, 2, 3])
            fn = ((1 - predictionBool) * label).sum(dim=[1, 2, 3])
            fp = (predictionBool * (~label)).sum(dim=[1, 2, 3])

            metrics[METRICS_LOSS_NDX, start_ndx:end_ndx] = diceLoss
            metrics[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics[METRICS_FP_NDX, start_ndx:end_ndx] = fp

        return diceLoss.mean() + fnLoss.mean() * 8 # return the weighted mean loss
        
    
    def doTraining(self, epoch_ndx, train_dl):
        """
        Iterates through the training batches and calls the CombuteBatchLoss for each.
        Then it updates the model parameters based on the loss. Also saves the train metrics
        and returns them. 
        """

        train_metrics = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        self.segmentation_model.train()
        train_dl.dataset.shuffleSamples()

        batch_ndx = 0
        for batch_tup in tqdm(train_dl, desc = 'Training Epoch {}/{}'.format(epoch_ndx, self.cli_args.epochs)):
            self.optimizer.zero_grad()
            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, train_metrics)
            loss_var.backwards()
            self.optimizer.step()
            batch_ndx += 1
        
        self.totalTrainingSamples_count += train_metrics.size(1)
        return train_metrics.to('cpu')
    

    def doValidation(self, epoch_ndx, val_dl):
        """Calculates metrics for the validation set"""
        with torch.no_grad():
            val_metrics = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)
            self.segmentation_model.eval()

            batch_ndx = 0
            for batch_tup in tqdm(val_dl, desc='Valdation on Epoch {}/{}'.format(epoch_ndx, self.cli_args.epochs)):
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, val_metrics)
                batch_ndx += 1 

        
        return val_metrics.to('cpu')
     
    def main(self):
    
        train_dl = self.initDataloader(False, 10, 3)
        val_dl = self.initDataloader(True, 10, 3)

        best_score = 0.0
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
    

            train_metrics = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', train_metrics)

            if epoch_ndx == 1 or epoch_ndx % 5 == 0:

                val_metrics = self.doValidation(epoch_ndx, val_dl)
                score = self.logMetrics(epoch_ndx, 'val', val_metrics)
                best_score = max(score, best_score)

                self.saveModel('seg', epoch_ndx, score == best_score)
        
        self.trn_writer.close()
        self.val_writer.close()
    

    def logMetrics(self, epoch_ndx, mode, metrics):
        metrics.detach().numpy()
        sum = metrics.sum(axis=1)
        allLabel_count = sum[METRICS_TP_NDX] + sum[METRICS_FN_NDX]
        
        metrics_dict={}
        metrics_dict['percent_all/tp'] = sum[METRICS_TP_NDX] / allLabel_count * 100
        metrics_dict['percent_all/fn'] = sum[METRICS_FN_NDX] / allLabel_count * 100
        metrics_dict['percent_all/fp'] = sum[METRICS_FP_NDX] / allLabel_count * 100

        precision = metrics_dict['pr/precision'] = sum[METRICS_TP_NDX] / (sum[METRICS_TP_NDX] + sum[METRICS_FP_NDX])
        recall = metrics_dict['pr/recall'] = sum[METRICS_TP_NDX] / (sum[METRICS_TP_NDX] + sum[METRICS_FN_NDX])

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / (precision + recall)

        print(("Epoch {}/{} {:8}"
             + "{loss/all:4f} loss"
             + "{pr/precision:.4f} precision"
             + "{pr/recall:.4f} recall"
             + "{pr/f1_score:.4f} F1 score").format(
                epoch_ndx,
                self.cli_args.epochs,
                mode,
                **metrics_dict,
             ))

        print(("{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9/1f}% fp"
        ).format(
            **metrics_dict,
        ))
       
        self.initTensorboardWriter()
        writer = getattr(self, mode + '_writer')
        

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamplles_count)
        
        writer.flush()

        score = metrics_dict['pr/recall']

        return score

    def saveModel(self, type, epoch, isBest):
        filepath = os.path.join(
            'models',
            self.cli_args.tb_prefix,
            '{}_{}_{}_{}.state'.format(
                type,
                self.time_str,
                self.cli_args.comment,
                self.totalTrainingSamples_count,
            )
        )

        os.makedirs(os.path.dirname(filepath), mode=0o775, exist_ok=True)

        model = self.segmentation_model
        
        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state' : self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }

        torch.save(state, filepath)

        print("Saved model params to {}".format(filepath))

        if isBest:
            best_path = os.path.join('models', 
                                     self.cli_args.tb_prefix,
                                     f'{type}_{self.time}_{self.cli_args.comment}.best_state')
            shutil.copyfile(filepath, best_path)

            print("Saved best model to {}".format(best_path))



if __name__ == '__main__':
    SegTraining().main()