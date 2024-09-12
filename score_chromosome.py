#!/usr/bin/env python
import argparse
import gc
import pathlib
import os
import numpy as np
import torch
from scoreUtils import *
from utils import *
from DconnNet import DconnNet
from multiprocessing import Pool, cpu_count
import torch.multiprocessing as mp
import resource


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description=" score every possible chromatin loop")

    parser.add_argument("-p", dest="path", type=str, default=None,
                        help="Path to a .cool URI string or a .hic file.")
    parser.add_argument("--balance",dest="balance", default= True,
                        help="Whether or not using the ICE/KR-balanced matrix.")
    parser.add_argument("-a", dest="bigwig",type=str, default=None,
                        help="Path to the chromatin accessibility data which is a bigwig file ")
    parser.add_argument("-o", dest="output", default='./scores/', help="Folder path to store results.")
    parser.add_argument("-m", dest="model", default=None, help="Path to a trained mode.")
    parser.add_argument('-l', '--lower', type=int, default=11,
                   help='''Lower bound of distance between loci in bins (default 2).''')
    parser.add_argument('-u', '--upper', type=int, default=300,
                   help='''Upper bound of distance between loci in bins (default 300).''')
    parser.add_argument('-w', '--width', type=int, default=11,
                   help='''Number of bins added to center of window. 
                            default width=11 corresponds to 23*23 windows''')
    parser.add_argument('-r', '--resolution',
                   help='Resolution in bp, default 10000',
                   type=int, default=10000)
    parser.add_argument('--minimum-prob', type=float, default=0.5,
                   help='''Only output pixels with probability score greater than this value (default 0.5)''')
    parser.add_argument('-c', dest="bigwig1",type=str, default=None,
                        help="Path to the ctcf chip-seq data which is a bigwig file")
    parser.add_argument('-q', dest='p2ll', type=float, default=None,
                        help="Threshold for mean_ll/center ratio ")
    parser.add_argument('-n', dest='processes', type=int, default=None,
                        help="Divide the program into multiple process chunks ")

    return parser.parse_args()


def set_file_descriptor_limit(limit=8192):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))

def main():
    set_file_descriptor_limit()

    args = get_args()
    np.seterr(divide='ignore', invalid='ignore')

    # pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model,map_location='cpu')
    state_dict = checkpoint["model_state_dict"]
    deepcnn = DconnNet()
    deepcnn.load_state_dict(state_dict)
    # deepcnn.to(torch.device("cuda:4"))
    deepcnn.to("cpu")

    # more robust to check if a file is .hic
    hic_info = read_hic_header(args.path)
    if hic_info is None:
        hic = False
    else:
        hic = True

    if not hic:
        import cooler
        Lib = cooler.Cooler(args.path)
        chromosomes = Lib.chromnames[:]
    else:
        chromosomes = get_hic_chromosomes(args.path, args.resolution)


    pre = find_chrom_pre(chromosomes)
    tmp = os.path.split(args.model)[1]
    # ccname is consistent with chromosome labels in .hic / .cool
    ccname = pre + tmp.split('_model')[0].lstrip('chr')
    cikada = 'chr' + ccname.lstrip('chr')  # cikada always has prefix "chr"

    # num_processes = min(4, len(chromosomes))
    # with Pool(num_processes) as p:
    #     p.starmap(process_chromosome, [(args, chrom, hic, deepcnn, cikada) for chrom in chromosomes])

    if not hic:
        M = Lib.matrix(balance=args.balance, sparse=True).fetch(ccname).tocsr()
        raw_M = Lib.matrix(balance=False, sparse=True).fetch(ccname).tocsr()
        # print(Lib.bins().fetch(ccname).columns)
        weights = Lib.bins().fetch(ccname)['weight'].values
        X = Chromosome(M, model=deepcnn, raw_M=raw_M, weights=weights, ATAC=args.bigwig, CTCF=args.bigwig1,
                        cname=cikada, lower=args.lower,
                        upper=args.upper, res=args.resolution,
                        width=args.width, p2ll=args.p2ll ,processes=args.processes)
    else:
        if args.balance:
            M = csr_contact_matrix('KR', args.path, ccname, ccname, 'BP', args.resolution)
            raw_M = csr_contact_matrix('NONE', args.path, ccname, ccname, 'BP', args.resolution)
            X = Chromosome(M, model=deepcnn, raw_M=raw_M, weights=None,  ATAC=args.bigwig, CTCF=args.bigwig1,
                                      cname=cikada, lower=args.lower,
                                      upper=args.upper, res=args.resolution,
                                      width=args.width, p2ll=args.p2ll ,processes=args.processes)
        else:
            M = csr_contact_matrix('NONE', args.path, ccname, ccname, 'BP', args.resolution)
            X = Chromosome(M, model=deepcnn, raw_M= M, weights=None,  ATAC=args.bigwig, CTCF=args.bigwig1,
                                      cname=cikada, lower=args.lower,
                                      upper=args.upper, res=args.resolution,
                                      width=args.width, p2ll=args.p2ll ,processes=args.processes)
    result, R = X.score(thre=args.minimum_prob)

    X.writeBed(args.output, result, R)

if __name__ == "__main__":
    main()