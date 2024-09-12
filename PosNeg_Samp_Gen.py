#!/usr/bin/env python
import pathlib
import hicstraw
import argparse
import numpy as np
from dataUtils import *
from utils import *

def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="generate positive and negative samples ")

    parser.add_argument("-p", dest="path", type=str, default=None,
                        help="Path to a .cool URI string or a .hic file.")
    parser.add_argument("--balance", dest="balance", default=True,
                        help="Whether or not using the ICE/KR-balanced matrix.")
    parser.add_argument('-b', '--bedpe',
                          help='''Path to the bedpe file containing positive training set.''')
    parser.add_argument('-c', dest="bigwig1", type=str,default=None,
                        help="ctcf CHIP-seq")
    parser.add_argument("-a", dest="bigwig", type=str, default=None,
                        help="Path to the chromatin accessibility data which is a bigwig file ")
    parser.add_argument("-o", dest="output", default='./data/', help="Folder path to store results.")
    parser.add_argument('-l', '--lower', type=int, default=2,
                        help='''Lower bound of distance between loci in bins (default 2).''')
    parser.add_argument('-u', '--upper', type=int, default=300,
                        help='''Upper bound of distance between loci in bins (default 300).''')
    parser.add_argument('-w', '--width', type=int, default=11,
                        help='''Number of bins added to center of window. 
                                default width=11 corresponds to 23*23 windows''')
    parser.add_argument('-r', '--resolution',
                        help='Resolution in bp, default 10000',
                        type=int, default=10000)


    return parser.parse_args()

def chrom_key(chrom_name):
    if chrom_name.startswith('chr'):
        name = chrom_name[3:]
    else:
        name = chrom_name
    if name.isdigit():
        return (int(name), chrom_name)
    elif name == 'X':
        return (23, chrom_name)
    elif name == 'Y':
        return (24, chrom_name)

    else:
        return (25, chrom_name)


def main():
    args = get_args()
    np.seterr(divide='ignore', invalid='ignore')

    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)

    # more robust to check if a file is .hic
    hic_info = read_hic_header(args.path)

    if hic_info is None:
        hic = False
    else:
        hic = True

    coords = parsebed(args.bedpe, lower=2, res=args.resolution)

    kde, lower, long_start, long_end = learn_distri_kde(coords)



    if not hic:
        import cooler
        Lib = cooler.Cooler(args.path)
        chromosomes = Lib.chromnames[:]
    else:
        chromosomes = get_hic_chromosomes(args.path, args.resolution)
    print(chromosomes)

    # train model per chromosome
    positive_class = {}
    positive_atac = {}
    positive_ctcf = {}
    negative_ctcf = {}
    negative_atac = {}
    negative_class = {}
    positive_labels = {}
    negative_labels = {}

    sorted_chromosomes = sorted(chromosomes, key=chrom_key)
    for key in sorted_chromosomes:
        if key.startswith('chr'):
            chromname = key
        else:
            chromname = 'chr'+key
        print('collecting from {}'.format(key))


    # for key in chromosomes:
    #     # if key != 'chr19':
    #     #     continue
    #     if key.startswith('chr'):
    #         chromname = key
    #     else:
    #         chromname = 'chr'+key
    #     print('collecting from {}'.format(key))

        if not hic:
            X = Lib.matrix(balance=True,
                           sparse=True).fetch(key).tocsr()

        else:
            if args.balance:
                X = csr_contact_matrix(
                    'KR', args.path, key, key, 'BP', args.resolution)
            else:
                X = csr_contact_matrix(
                    'NONE', args.path, key, key, 'BP', args.resolution)
        clist = coords[chromname]



        #####generate positive samples
        try:
            positive_results = list(generateHIC(X, clist, chromname, files=args.bigwig, resou=args.resolution, width=args.width,
                                       random_seed=42))
            positive_class[chromname] = np.array([f[0].tolist() for f in positive_results])
            used_positive_coords = [f[1] for f in positive_results]
            print(len(positive_class[chromname]))
            positive_class[chromname] = positive_class[chromname].reshape((positive_class[chromname].shape[0], 1,
                                                                               positive_class[chromname].shape[1],
                                                                               positive_class[chromname].shape[2]))

            positive_atac[chromname] = np.array([f.tolist() for f in generateATAC(
                 used_positive_coords, chromname, files=args.bigwig, resou=args.resolution,  width=args.width)])
            print(len(positive_atac[chromname]))
            positive_atac[chromname] = positive_atac[chromname].reshape((positive_atac[chromname].shape[0], 1,
                                                                         positive_atac[chromname].shape[1],
                                                                         positive_atac[chromname].shape[2]))

            positive_ctcf[chromname] = np.array([f.tolist() for f in generateProtein(
                used_positive_coords, chromname, files=args.bigwig1, resou=args.resolution, width=args.width)])
            print(len(positive_ctcf[chromname]))
            positive_ctcf[chromname] = positive_ctcf[chromname].reshape((positive_ctcf[chromname].shape[0], 1,
                                                                         positive_ctcf[chromname].shape[1],
                                                                         positive_ctcf[chromname].shape[2]))

            positive_num = len((positive_class[chromname]))
            positive_labels[chromname] = np.ones(positive_num).tolist()


            neg_coords = negative_generating(X, kde, clist, lower, long_start, long_end)
            stop = 5 * positive_num
            negative_results = list(generateHIC(
                X, neg_coords, chromname, files=args.bigwig, resou=args.resolution, width=args.width, positive=False, stop=stop, random_seed=42))
            negative_class[chromname] = np.array([f[0].tolist() for f in negative_results])
            used_negative_coords = [f[1] for f in negative_results]
            print(len(negative_class[chromname]))
            negative_class[chromname] = negative_class[chromname].reshape((negative_class[chromname].shape[0], 1,
                                                                           negative_class[chromname].shape[1], negative_class[chromname].shape[2]))

            negative_atac[chromname] = np.array([f.tolist() for f in generateATAC(
                used_negative_coords, chromname, files=args.bigwig, resou=args.resolution, width=args.width)])
            print(len(negative_atac[chromname]))
            negative_atac[chromname] = negative_atac[chromname].reshape((negative_atac[chromname].shape[0], 1,
                                                                         negative_atac[chromname].shape[1],
                                                                         negative_atac[chromname].shape[2]))

            negative_ctcf[chromname] = np.array([f.tolist() for f in generateProtein(
                used_negative_coords, chromname, files=args.bigwig1, resou=args.resolution, width=args.width)])
            print(len(negative_ctcf[chromname]))
            negative_ctcf[chromname] = negative_ctcf[chromname].reshape((negative_ctcf[chromname].shape[0], 1,
                                                                         negative_ctcf[chromname].shape[1],
                                                                         negative_ctcf[chromname].shape[2]))



            negative_num = len(negative_class[chromname])
            negative_labels[chromname] = np.zeros(negative_num).tolist()

            np.savez(args.output+'%s_positive.npz' % chromname, data=positive_class[chromname],
                     atac=positive_atac[chromname],ctcf=positive_ctcf[chromname], label=positive_labels[chromname])
            np.savez(args.output+'%s_negative.npz' % chromname, data=negative_class[chromname],
                     atac=negative_atac[chromname],ctcf=negative_ctcf[chromname], label=negative_labels[chromname])


        except:
            print(chromname, ' failed to gather fts')



if __name__ == "__main__":
    main()
