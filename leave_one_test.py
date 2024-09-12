#!/usr/bin/python
# coding=utf-8

import os
import sys
import time
import argparse
import random
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from DconnNet import DconnNet
from datasets import EPIDataSetTrain, EPIDataSetTest
from trainer import Trainer
from loss import OhemLoss
from utils import *
from sklearn.metrics import f1_score,precision_recall_curve,auc, matthews_corrcoef
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import recall_score
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import pandas as pd


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="FCN for motif location")

    parser.add_argument("-d", dest="data_dir", type=str, default=None,
                        help="A directory containing the training data.")

    parser.add_argument("-g", dest="gpu", type=str, default='3',
                        help="choose gpu device.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/',
                        help="Where to save snapshots of the model.")
    parser.add_argument("-f", dest="file_path", type=str, required=True,
                        help="The name of the file to be opened (without extension).")

    return parser.parse_args()


def main():
    """Create the model and start the training."""
    args = get_args()
    if torch.cuda.is_available():
        if len(args.gpu.split(',')) == 1:
            device = torch.device("cuda:" + args.gpu)
        else:
            device = torch.device("cuda:" + args.gpu.split(',')[0])
    else:
        device = torch.device("cpu")

    files = os.listdir(args.data_dir)
    chromnames = []
    for file in files:
        ccname = file.split('_')[0]
        if ccname not in chromnames:
            chromnames.append(ccname)
    f = open(args.file_path, 'w')
    f.write('chromname,f1-score,prauc,precision,recall\n')

    results = []  # List to store results
    chromnames1 = ['chr15', 'chr16', 'chr17']
    for key in chromnames1:
        # if key == 'chr17':
            Data = np.load(osp.join(args.data_dir, '%s_negative.npz' % key))
            seqs_neg, atat_neg, label_neg, ctcf_neg = Data['data'], Data['atac'], Data['label'], Data['ctcf']
            # seqs_neg,  label_neg = Data['data'], Data['label']

            Data = np.load(osp.join(args.data_dir, '%s_positive.npz' % key))
            seqs_pos, atac_pos, label_pos, ctcf_pos = Data['data'], Data['atac'], Data['label'], Data['ctcf']
            # seqs_pos,  label_pos, = Data['data'], Data['label']

            seqs = np.concatenate((seqs_pos, seqs_neg), axis=0)
            atac = np.concatenate((atac_pos, atat_neg), axis=0)
            ctcf = np.concatenate((ctcf_pos, ctcf_neg), axis=0)
            label = np.concatenate((label_pos, label_neg), axis=0)

        # histone1 = np.concatenate((histone1_pos,histone1_neg),axis=0)
            for i in range(len(atac)):
                atac[i] = np.log10(1 + atac[i]*10)
                seqs[i] = np.log10(1 + seqs[i]*10)
                ctcf[i] = np.log10(1 + ctcf[i]*10)

                atac[i] = atac[i]/np.max(atac[i]+1)
                seqs[i] = seqs[i]/np.max(seqs[i]+1)
                ctcf[i] = ctcf[i]/np.max(ctcf[i]+1)
            if seqs.shape[0] != atac.shape[0] or seqs.shape[0] != atac.shape[0]:
                data_arrays = [seqs, atac, ctcf, label]
                # data_arrays = [seqs, atac, label]
                min_length = min([arr.shape[0] for arr in data_arrays])
                data_arrays = [arr[:min_length] for arr in data_arrays]
                seqs, atac, ctcf, label = data_arrays
                # seqs, atac,  label = data_arrays

            seqs = np.concatenate((seqs, atac, ctcf), axis=1)
            # seqs = np.concatenate((seqs, atac), axis=1)

        # Balanced test set
            num = np.arange(0, len(label)).reshape(-1, 1)
            ros = RandomUnderSampler(random_state=42)
            num_resampled, label_resampled = ros.fit_resample(num, label)
            num_resampled = np.squeeze(num_resampled)
            seqs_resampled = seqs[num_resampled]
            test_data_balanced = EPIDataSetTest(seqs_resampled, label_resampled)
            test_loader_balanced = DataLoader(test_data_balanced, batch_size=1, shuffle=False, num_workers=1)
        # build test data generator
        #     data_te = seqs
        #     label_te = label
        #     test_data = EPIDataSetTest(data_te, label_te)
        #     test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

        # Load weights
            checkpoint_file = osp.join(args.checkpoint, '{}_model_best.pth'.format(key))
            chk = torch.load(checkpoint_file)
            state_dict = chk['model_state_dict']
            model = DconnNet()

            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            torch.backends.cudnn.benchmark = True
            p_all = []
            t_all = []
            for i_batch, sample_batch in enumerate(test_loader_balanced):
                X_data = sample_batch["data"].float().to(device)
                signal = sample_batch["label"].float()
                with torch.no_grad():
                    pred = model(X_data)
                p_all.append(pred.view(-1).data.cpu().numpy()[0])
                t_all.append(signal.view(-1).data.numpy()[0])
            f1 = f1_score(t_all, [int(x > 0.5) for x in p_all])
            # precision, recall, _ = precision_recall_curve(t_all, [int(x > 0.5) for x in p_all])
            precision, recall, _ = precision_recall_curve(t_all, p_all)
            prauc = auc(recall, precision)
            mcc = matthews_corrcoef(t_all, [int(x > 0.5) for x in p_all])
            ap = average_precision_score(t_all, [int(x > 0.5) for x in p_all])
            rscore = recall_score(t_all, [int(x > 0.5) for x in p_all])

            results.append([key, precision, recall, prauc, f1, mcc])

            f.write("chrom: {}\tf1: {:.3f}\tprauc: {:.3f}\tMCC: {:.3f}\tap: {:.3f}\trscore: {:.3f}\n".format(key, f1, prauc, mcc, ap, rscore))
    f.close()


    plt.rcParams.update({'font.size': 16})

    # Plotting Precision-Recall curves
    plt.figure(figsize=(10, 7))
    for key, precision, recall, prauc, f1, mcc in results:
        plt.plot(recall, precision, label=f'{key} PRAUC={prauc:.3f}, F1={f1:.3f}, MCC={mcc:.3f}')

    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.title(' Precision-Recall Curve on 90% Downsampled (1800M) GM12878 Cells  ', fontsize=20) # Using HIC And ChIP-seq Data  # MESC Cells
    plt.legend(loc='best', fontsize=14)
    plt.grid(True)

    # Save the plot to a file
    save_directory = os.path.dirname(args.file_path.rstrip('/'))
    save_path = os.path.join(save_directory, 'precision_recall_curve.png')
    plt.savefig(save_path)
    # plt.show()

if __name__ == "__main__":
    main()

