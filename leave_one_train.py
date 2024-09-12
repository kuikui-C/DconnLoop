#!/usr/bin/python

import os
import sys
import time
import argparse
import random
import numpy as np
import os.path as osp


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from imblearn.under_sampling import RandomUnderSampler
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, precision_recall_curve, auc
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score, balanced_accuracy_score


from DconnNet import DconnNet
from datasets import EPIDataSetTrain, EPIDataSetTest, EPIDataSetVal
from trainer import Trainer
from loss import OhemLoss
from collections import OrderedDict



def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description=" DLoopCaller train model for chromatin loops")

    parser.add_argument("-d", dest="data_dir", type=str, default=None,
                        help="A directory containing the training data.")
    parser.add_argument("-g", dest="gpu", type=str, default='1',
                        help="choose gpu device. eg. '0,1,2' ")
    parser.add_argument("-s", dest="seed", type=int, default=5,
                        help="Random seed to have reproducible results.")
    # Arguments for Adam or SGD optimization
    parser.add_argument("-b", dest="batch_size", type=int, default=1,
                        help="Number of sequences sent to the network in one step.")
    parser.add_argument("-lr", dest="learning_rate", type=float, default=0.01,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("-m", dest="momentum", type=float, default=0.9,
                        help="Momentum for the SGD optimizer.")
    parser.add_argument("-e", dest="max_epoch", type=int, default=30,
                        help="Number of training steps.")
    parser.add_argument("-w", dest="weight_decay", type=float, default=0.0005,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("-p", dest="power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")

    parser.add_argument("-r", dest="restore", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/',
                        help="Where to save snapshots of the model.")
    parser.add_argument("-nc", dest="num_classes", type=int, default=2)

    return parser.parse_args()


def main():
    """Create the model and start the training."""
    args = get_args()
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        if len(args.gpu.split(',')) == 1:
            device = torch.device("cuda:" + args.gpu)
        else:
            device = torch.device("cuda:" + args.gpu.split(',')[0])
    else:
        device = torch.device("cpu")
        torch.cuda.manual_seed_all(args.seed)


    files = os.listdir(args.data_dir)
    chromnames = []
    for file in files:
        ccname = file.split('_')[0]
        if ccname not in chromnames:
            chromnames.append(ccname)

    chromnames1 = [ 'chr15', 'chr16','chr17']
    # test_chrom = 'chr17'
    #  for test_chrom in chromnames:
    for test_chrom in chromnames1:
        print(test_chrom)
        seqs = []
        label = []
        atac_info = []
        ctcf_info = []
        chromname = []

        for filename in files:
            if filename.split('_')[0] != test_chrom:
                Data = np.load(osp.join(args.data_dir, filename))
                chrom = filename.split('_')[0]
                num_samples = len(Data['label'])
                seqs.extend(Data['data'])
                atac_info.extend(Data['atac'])
                ctcf_info.extend(Data['ctcf'])
                label.extend(Data['label'])
                chromname.extend([chrom] * num_samples)

        seqs = np.array(seqs)
        label = np.array(label)
        atac_info = np.array(atac_info)
        ctcf_info = np.array(ctcf_info)
        label = label.reshape((label.shape[0], 1))
        chromname = np.array(chromname)

        data_arrays = [seqs, atac_info, ctcf_info, label]
        min_length = min([arr.shape[0] for arr in data_arrays])
        data_arrays = [arr[:min_length] for arr in data_arrays]
        seqs, atac_info, ctcf_info, label = data_arrays
        print(seqs.shape)
        print(label.shape)


        for i in range(len(atac_info)):
            # atac_info[i] = np.arcsinh(atac_info[i] * 10)
            atac_info[i] = np.log10(1 + atac_info[i]*10)
            atac_info[i] = atac_info[i]/np.max(atac_info[i]+1)

            # ctcf_info[i] = np.arcsinh(ctcf_info[i] * 10)
            ctcf_info[i] = np.log10(1 + ctcf_info[i] * 10)
            ctcf_info[i] = ctcf_info[i] / np.max(ctcf_info[i] + 1)

            # seqs[i] = np.arcsinh(seqs[i] * 10)
            seqs[i] = np.log10(1 + seqs[i] * 10)
            seqs[i] = seqs[i]/np.max(seqs[i]+1)

        # Training set
        data = np.concatenate((seqs, atac_info, ctcf_info), axis=1)
        index_train = list(range(len(data)))
        label_train = label[index_train]
        print("data_train:", data.shape)

        # Divide into positive and negative
        num_pos = np.sum(label_train == 1)
        num_neg = np.sum(label_train == 0)
        print(num_pos, num_neg)
        ratio = round(num_neg / num_pos)
        index_pos = np.where(label_train == 1)
        index_neg = np.where(label_train == 0)
        print('ratio:', ratio)

        # Positive data
        data_train_pos = data[index_pos[0], :, :, :]
        label_train_pos = label_train[index_pos]
        print("data_train_pos", data_train_pos.shape)

        # Negative data
        data_train_neg_total = data[index_neg[0], :, :, :]
        label_train_neg_total = label_train[index_neg]
        print("data_train_neg_total:", data_train_neg_total.shape)

        f1_best = 0
        kf = KFold(ratio, shuffle=True, random_state=0)     # The ratio parameter indicates how many folds the data set is divided into
        for _, index in kf.split(label_train_neg_total):    # Each fold provides a different index
            data_train_neg = data_train_neg_total[index]
            label_train_neg = label_train_neg_total[index]
            print("data_train_neg:",data_train_neg.shape)

            data_train_kf = np.concatenate((data_train_pos, data_train_neg), axis=0)
            label_train_kf = np.concatenate((label_train_pos, label_train_neg))
            print("data_train_kf:",data_train_kf.shape)

            # Divide training set and validation set
            data_train_kf, data_train_kf_val_kf, label_train_kf, label_val_kf = train_test_split(
                data_train_kf,
                label_train_kf,
                test_size=0.2,
                random_state=0,
                )
            print("data_train_kf_val_kf:",data_train_kf_val_kf.shape)

            data_tr = data_train_kf
            label_tr = label_train_kf
            train_data = EPIDataSetTrain(data_tr, label_tr)
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)

            data_va = data_train_kf_val_kf
            label_va = label_val_kf
            val_data = EPIDataSetVal(data_va, label_va)
            val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

            model = DconnNet()
            optimizer = optim.Adam(model.parameters(),
                                   lr=args.learning_rate)
            criterion = OhemLoss()
            start_epoch = 0

            if args.restore:
                print("Resume it from {}.".format(args.restore_from))
                checkpoint = torch.load(args.restore)
                state_dict = checkpoint["model_state_dict"]
                model.load_state_dict(state_dict,
                                      strict=True)

            # if there exists multiple GPUs, using DataParallel
            if len(args.gpu.split(',')) > 1 and (torch.cuda.device_count() > 1):
                model = nn.DataParallel(model, device_ids=[int(id_) for id_ in args.gpu.split(',')])

            executor = Trainer(model=model,
                               optimizer=optimizer,
                               criterion=criterion,
                               device=device,
                               checkpoint=args.checkpoint,
                               start_epoch=start_epoch,
                               max_epoch=args.max_epoch,
                               train_loader=train_loader,
                               test_loader=None,
                               val_loader=val_loader,
                               lr_policy='CosineAnnealingWarmRestarts',              # # 'CosineAnnealingWarmRestarts'  'poly'  'step'  'warm-up-epoch'  'warm-up-step'
                               early_stopping_monitor=5)
            r, f1, state_dict, accuracy, precision = executor.train()

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # Remove 'module.' prefix
                new_state_dict[name] = v

            if (f1_best) < (f1):
                print("Store the weights of the model in the current run.\n")

                f1_best = f1
                checkpoint_file = osp.join(args.checkpoint, '{}_model_best.pth'.format(test_chrom))
                torch.save({'model_state_dict':  new_state_dict}, checkpoint_file)


if __name__ == "__main__":
    main()
