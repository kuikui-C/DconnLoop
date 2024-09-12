import os
import gc
import pathlib
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats
import pyBigWig
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from tqdm import tqdm
from dataUtils import distance_normaize_core, calculate_expected
from statsmodels.stats.multitest import multipletests
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, Manager
from threading import Thread
import multiprocessing as mp
from functools import partial


class Chromosome():
    def __init__(self, M, model, raw_M= None, weights=None, ATAC= None, CTCF= None, lower=11, upper=300, cname='chrm', res=10000, width=11, p2ll=0.6, processes=12 ):
        # cLen = coomatrix.shape[0] # seems useless
        R, C = M.nonzero()

        validmask = np.isfinite(M.data) & (C-R > (-2*width)) & (C-R < (upper+2*width))

        R, C, data = R[validmask], C[validmask], M.data[validmask]
        self.M = sparse.csr_matrix((data, (R, C)), shape=M.shape)
        if weights is None:
            self.exp_arr = calculate_expected(M, upper + 2 * width, raw=True)
            if M is raw_M:
                self.background = self.exp_arr
            else:
                self.background = calculate_expected(raw_M, upper + 2 * width, raw=True)
        else:
            self.exp_arr = calculate_expected(M, upper + 2 * width, raw=False)
            self.background = self.exp_arr
        lower = max(lower, width + 1)
        upper = min(upper, M.shape[0] - 2 * width)
        self.raw_M = raw_M
        self.weights = weights
        self.get_candidate(lower, upper)
        self.ATAC = ATAC
        self.CTCF = CTCF
        self.chromname = cname
        self.r = res
        self.w = width
        self.model = model
        self.p2ll = p2ll
        self.processes = processes

    def get_candidate(self, lower, upper):

        x_arr = np.array([], dtype=int)
        y_arr = np.array([], dtype=int)
        p_arr = np.array([], dtype=float)
        idx = np.arange(self.raw_M.shape[0])
        for i in range(lower, upper + 1):
            diag = self.raw_M.diagonal(i)
            e = self.background[i]
            if (diag.size > 0) and (e > 0):
                xi = idx[:-i]
                yi = idx[i:]
                if self.weights is None:
                    exp = np.ones(diag.size, dtype=float) * e
                else:
                    b1 = self.weights[:-i]
                    b2 = self.weights[i:]
                    exp = np.ones(diag.size, dtype=float) * e / (b1 * b2)

                Poiss = stats.poisson(exp)
                pvalues = Poiss.sf(diag)
                mask = (diag > 0) & np.isfinite(pvalues)
                x_arr = np.r_[x_arr, xi[mask]]
                y_arr = np.r_[y_arr, yi[mask]]
                p_arr = np.r_[p_arr, pvalues[mask]]

        qvalues = multipletests(p_arr, method = 'fdr_bh')[1]
        mask = p_arr < 0.01
        # mask = qvalues < 0.05
        self.ridx, self.cidx = x_arr[mask], y_arr[mask]

    def read_and_process_bigwig(self, bw_file, chromname, start, end, width, resolution):
        data = np.array(bw_file.values(chromname, start * resolution, end * resolution))
        data[np.isnan(data)] = 0
        return data.reshape(2 * width + 1, resolution)

    def process_window(self, window, x, y, width):

        center = window[width, width]
        mean_ll = np.mean(window[-width:, :width])
        if mean_ll > 0 and (center / mean_ll) > self.p2ll:

            window = distance_normaize_core(window, self.exp_arr, x, y, width)
            window = gaussian_filter(window, sigma=1, order=0)
            return window
        return None

    def process_chunk(self,chunk, chunk_index, total_chunks):
        seq, clist, atac, ctcf = [], [], [], []
        width = self.w

        # bw = pyBigWig.open(self.ATAC)
        # bw1 = pyBigWig.open(self.CTCF)

        total_coordinates = 0
        processed_coordinates = 0
        with pyBigWig.open(self.ATAC) as bw, pyBigWig.open(self.CTCF) as bw1:
            for i, c in enumerate(chunk):
                total_coordinates += 1
                # if (i + 1) % 1000 == 0:
                #     print(f"Processed {i + 1} coordinates in chunk {chunk_index + 1}.")
                x, y = c[0], c[1]
                if all([x - width >= 0, x + width + 1 <= self.M.shape[1], y - width >= 0,
                        y + width + 1 <= self.M.shape[0]]):
                    try:
                        window = self.M[x - width:x + width + 1, y - width:y + width + 1].toarray()
                    except:
                        continue
                    window[np.isnan(window)] = 0
                    if np.count_nonzero(window) < window.size * .1:
                        continue

                    window = self.process_window(window, x, y, width)
                    if window is None:
                        continue

                    if np.isfinite(window).all() and window.shape == (2 * width + 1, 2 * width + 1):
                        try:
                            window_x = self.read_and_process_bigwig(bw, self.chromname, x - width, x + width + 1, width,
                                                                    self.r)
                            window_x1 = self.read_and_process_bigwig(bw1, self.chromname, x - width, x + width + 1, width,
                                                                     self.r)

                            bilateral_data_x = [self.bilateral(row) for row in window_x]
                            window_x = [np.array(bilateral_data_x)]

                            bilateral_data_x1 = [self.bilateral(row) for row in window_x1]
                            window_x1 = [np.array(bilateral_data_x1)]

                            window_y = self.read_and_process_bigwig(bw, self.chromname, y - width, y + width + 1, width,
                                                                    self.r)
                            window_y1 = self.read_and_process_bigwig(bw1, self.chromname, y - width, y + width + 1, width,
                                                                     self.r)

                            bilateral_data_y = [self.bilateral(row) for row in window_y]
                            window_y = [np.array(bilateral_data_y)]

                            bilateral_data_y1 = [self.bilateral(row) for row in window_y1]
                            window_y1 = [np.array(bilateral_data_y1)]

                            window_atac = np.dot(np.transpose(window_x), window_y)
                            window_atac = gaussian_filter(window_atac, sigma=1, order=0)
                            window_ctcf = np.dot(np.transpose(window_x1), window_y1)
                            window_ctcf = gaussian_filter(window_ctcf, sigma=1, order=0)

                            seq.append(window)
                            clist.append(c)
                            atac.append(window_atac)
                            ctcf.append(window_ctcf)

                            processed_coordinates += 1
                        except:
                            continue

        # bw.close()
        # bw1.close()
        print(f"Finished processing chunk {chunk_index + 1} of {total_chunks}, processed {total_coordinates} coordinates, filtered {processed_coordinates} coordinates")
        return seq, clist, atac, ctcf

    def getwindow(self, coords):
        num_chunks = self.processes  # Number of chunks/processes
        chunk_size = len(coords) // num_chunks
        chunks = [coords[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

        if len(coords) % num_chunks != 0:
            chunks[-1].extend(coords[num_chunks * chunk_size:])

        total_chunks = len(chunks)

        with Pool(num_chunks) as pool:
            results = pool.starmap(partial(self.process_chunk, total_chunks=total_chunks),
                                   [(chunk, i) for i, chunk in enumerate(chunks)])

        seq, clist, atac, ctcf = [], [], [], []
        for res_seq, res_clist, res_atac, res_ctcf in results:
            seq.extend(res_seq)
            clist.extend(res_clist)
            atac.extend(res_atac)
            ctcf.extend(res_ctcf)

        seq = np.array(seq)
        atac = np.array(atac)
        ctcf = np.array(ctcf)

        seq = seq.reshape((seq.shape[0], 1, seq.shape[1], seq.shape[2]))
        atac = atac.reshape((atac.shape[0], 1, atac.shape[1], atac.shape[2]))
        ctcf = ctcf.reshape((ctcf.shape[0], 1, ctcf.shape[1], ctcf.shape[2]))

        for i in range(len(seq)):
            seq[i] = seq[i] / np.max(seq[i] + 1)
            atac[i] = np.log10(1 + atac[i] * 10)
            atac[i] = atac[i] / np.max(atac[i] + 1)
            ctcf[i] = np.log10(1 + ctcf[i] * 10)
            ctcf[i] = ctcf[i] / np.max(ctcf[i] + 1)

        fts = np.concatenate((seq, atac, ctcf), axis=1)
        return fts, clist

    def test(self, fts):
        num_total = len(fts)
        batch = 10000
        iteration = int(np.ceil(num_total/batch))
        preds = np.array([])
        for i in range(iteration):
            segment = fts[i*batch:(i+1)*batch]
            segment = torch.from_numpy(segment)
            # segment = segment.float().to(torch.device("cuda:4"))
            segment = segment.float().to(torch.device("cpu"))
            # segment = segment.float().to('cuda')
            with torch.no_grad():
                label_p = self.model(segment)
            probas = label_p.view(-1).data.cpu().numpy()
            preds = np.concatenate((preds, probas))
            print("The current number is {}".format((i+1)*batch))

        return preds




    def score(self, thre=0.5):
        print('scoring matrix {}'.format(self.chromname))
        print('num candidates {}'.format(self.ridx.size))
        coords = [(r, c) for r, c in zip(self.ridx, self.cidx)]
        fts, clist = self.getwindow(coords)
        p = self.test(fts)
        clist = np.r_[clist]
        pfilter = p > thre
        ri = clist[:, 0][pfilter]
        ci = clist[:, 1][pfilter]
        result = sparse.csr_matrix((p[pfilter], (ri, ci)), shape=self.M.shape)
        data = np.array(self.M[ri, ci]).ravel()
        self.M = sparse.csr_matrix((data, (ri, ci)), shape=self.M.shape)

        return result, self.M

    def writeBed(self, out, prob_csr, raw_csr):
        pathlib.Path(out).mkdir(parents=True, exist_ok=True)
        with open(out + '/' + self.chromname + '.bed', 'w') as output_bed:
            r, c = prob_csr.nonzero()
            for i in range(r.size):
                line = [self.chromname, r[i]*self.r, (r[i]+1)*self.r,
                        self.chromname, c[i]*self.r, (c[i]+1)*self.r,
                        prob_csr[r[i],c[i]], raw_csr[r[i],c[i]]]
                output_bed.write('\t'.join(list(map(str, line)))+'\n')

    def bilateral(self, data, gauss_scaling=100, bilat_scaling=10, epsilon=1e-3):

        data_std = np.std(data) + epsilon
        gauss_width = data_std * gauss_scaling
        bilat_width = data_std * bilat_scaling

        smooth_data = gaussian_filter1d(data, gauss_width) + epsilon

        diff = np.abs(data - smooth_data)

        weights = gaussian_filter1d(diff, bilat_width) + epsilon

        bilateral_data = data * weights
        weights_sum = np.sum(weights)

        if weights_sum < 0.01:
            return data

        bilateral_data = np.sum(bilateral_data) / weights_sum
        # print(bilateral_data)
        return bilateral_data
