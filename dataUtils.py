#!/usr/bin/env python
# Program to train classifier given a cooler file and
# paired bedfile containing ChIA-PET peaks
# Author: Tarik Salameh

import numpy as np
from collections import defaultdict, Counter
from scipy import stats
import random
import pyBigWig
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csr_matrix
from sklearn.isotonic import IsotonicRegression
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
from threading import Thread
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures


def parsebed(chiafile, res=10000, lower=1, upper=5000000):

    coords = defaultdict(set)
    upper = upper // res
    with open(chiafile) as o:
        for line in o:
            s = line.rstrip().split()
            a, b = float(s[1]), float(s[4])
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            a //= res
            b //= res
            # all chromosomes including X and Y
            if (b-a > lower) and (b-a < upper) and 'M' not in s[0]:
                # always has prefix "chr", avoid potential bugs
                chrom = 'chr' + s[0].lstrip('chr')
                coords[chrom].add((a, b))

    for c in coords:
        coords[c] = sorted(coords[c])

    return coords


def learn_distri_kde(coords):

    dis = []
    for c in coords:
        for a, b in coords[c]:
            dis.append(b-a)

    lower = min(dis)

    # part 1: same distance distribution as the positive input
    kde = stats.gaussian_kde(dis)

    # part 2: random long-range interactions
    counts, bins = np.histogram(dis, bins=100)
    long_end = int(bins[-1])
    tp = np.where(np.diff(counts) >= 0)[0] + 2

    long_start = int(bins[tp[0]])

    return kde, lower, long_start, long_end


def negative_generating(M, kde, positives, lower, long_start, long_end):

    positives = set(positives)
    N = 5 * len(positives)
    # part 1: kde trained from positive input
    part1 = kde.resample(N).astype(int).ravel()
    part1 = part1[(part1 >= lower) & (part1 <= long_end)]

    # part 2: random long-range interactions
    part2 = []
    pool = np.arange(long_start, long_end+1)
    tmp = np.cumsum(M.shape[0]-pool)
    ref = tmp / tmp[-1]
    for i in range(N):
        r = np.random.random()
        ii = np.searchsorted(ref, r)
        part2.append(pool[ii])

    sample_dis = Counter(list(part1) + part2)

    neg_coords = []
    midx = np.arange(M.shape[0])

    for i in sorted(sample_dis):
        n_d = sample_dis[i]
        R, C = midx[:-i], midx[i:]
        tmp = np.array(M[R, C]).ravel()
        tmp[np.isnan(tmp)] = 0
        mask = tmp > 0
        R, C = R[mask], C[mask]
        pool = set(zip(R, C)) - positives
        sub = random.sample(pool, n_d)
        neg_coords.extend(sub)

    random.shuffle(neg_coords)

    return neg_coords


def getbigwig(file,chrome,start,end):
    bw = pyBigWig.open(file)
    sample = np.array(bw.values(chrome,start,end))
    sample[np.isnan(sample)] = 0
    bw.close()
    return sample

def generateHIC(Matrix, coords, chromname, files, resou, width=11, positive=True, stop=5000, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    negcount = 0
    coords = np.array(coords)
    a, b = coords[:, 0], coords[:, 1]
    # bounds checking
    valid_mask = (a - width >= 0) & (a + width + 1 <= Matrix.shape[1]) & (b - width >= 0) & (b + width + 1 <= Matrix.shape[0]) & (b - a > width)
    valid_x = a[valid_mask]
    valid_y = b[valid_mask]
    if valid_x.size < 10:
        return

    try:
        maxdis = max([abs(i - j) for i, j in zip(valid_x, valid_y)]) + 2 * width
        exp_arr = calculate_expected(Matrix, maxdis)
    except Exception as e:
        print(f"Error calculating expected values: {e}")
        return

    pbar = tqdm(total=0, desc='Processing HIC Data', leave=True)
    output_count = 0

    try:
        for x, y in zip(valid_x, valid_y):
            try:
                window = Matrix[x - width:x + width + 1, y - width:y + width + 1].toarray()
                if window.size != 23 * 23:
                    continue
                window[np.isnan(window)] = 0
                if np.count_nonzero(window) < window.size * .1:
                    continue

                center = window[width, width]
                mean_value = np.mean(window[-width:, :width])
                p2LL = center / mean_value if mean_value != 0 else float('inf')
                if positive and p2LL < 0.1:
                    continue

                window = distance_normaize_core(window, exp_arr, x, y, width)
                window = gaussian_filter(window, sigma=1, order=0)

                if np.all(np.isfinite(window)):
                    if not positive:
                        negcount += 1
                    if negcount > stop:
                        raise StopIteration
                    output_count += 1
                    pbar.total = output_count
                    pbar.update(1)
                    yield window, (x, y)
            except:
                continue
    finally:
        pbar.close()

def generateATAC_worker(params):
    chunk, chromname, files, resou, width, progress_queue = params
    results = []
    for x, y in chunk:
        try:
            window_x = getbigwig(files, chromname, (x - width) * resou, (x + width + 1) * resou)
            if window_x.size != 23 * 10000:
                continue

            window_x = window_x.reshape(23, 10000)
            bilateral_data_x = [bilateral(row) for row in window_x]
            window_x = [np.array(bilateral_data_x)]

            window_y = getbigwig(files, chromname, (y - width) * resou, (y + width + 1) * resou)
            if window_y.size != 23 * 10000:
                continue

            window_y = window_y.reshape(23, 10000)
            bilateral_data_y = [bilateral(row) for row in window_y]
            window_y = [np.array(bilateral_data_y)]

            window_atac = np.dot(np.transpose(window_x), window_y)
            window_atac = gaussian_filter(window_atac, sigma=1, order=0)

            if window_atac.shape == (23, 23):
                results.append(window_atac)
        except:
            continue
        finally:
            progress_queue.put(1)

    return results


def generateATAC(coords, chromname, files, resou, width=11, num_workers=36):
    manager = Manager()
    progress_queue = manager.Queue()

    coords = np.array(coords)
    valid_x, valid_y = coords[:, 0], coords[:, 1]

    chunk_size = len(valid_x) // num_workers
    chunks = [list(zip(valid_x, valid_y))[i:i + chunk_size] for i in range(0, len(valid_x), chunk_size)]
    params = [(chunk, chromname, files, resou, width, progress_queue) for chunk in chunks]

    total_tasks = len(valid_x)
    pbar = tqdm(total=total_tasks, desc='Processing ATAC Data', leave=True)

    def update_progress():
        while True:
            item = progress_queue.get()
            if item is None:
                break
            pbar.update(1)

    # Start the progress updater thread
    progress_thread = Thread(target=update_progress)
    progress_thread.start()

    final_results = []
    with Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(generateATAC_worker, params):
            for matrix in result:
                if matrix.shape == (23, 23):
                    final_results.append(matrix)
        progress_queue.put(None)

    progress_thread.join()
    pbar.close()

    return final_results



def generateProtein_worker(params):
    chunk, chromname, files, resou, width, progress_queue = params
    results = []
    for x, y in chunk:
        try:
            window_x = getbigwig(files, chromname, (x - width) * resou, (x + width + 1) * resou)
            if window_x.size != 23 * 10000:
                continue

            window_x = window_x.reshape(23, 10000)
            bilateral_data_x = [bilateral(row) for row in window_x]
            window_x = [np.array(bilateral_data_x)]

            window_y = getbigwig(files, chromname, (y - width) * resou, (y + width + 1) * resou)
            if window_y.size != 23 * 10000:
                continue

            window_y = window_y.reshape(23, 10000)
            bilateral_data_y = [bilateral(row) for row in window_y]
            window_y = [np.array(bilateral_data_y)]

            window_ctcf = np.dot(np.transpose(window_x), window_y)
            window_ctcf = gaussian_filter(window_ctcf, sigma=1, order=0)

            if window_ctcf.shape == (23, 23):
                results.append(window_ctcf)
        except:
            continue
        finally:
            progress_queue.put(1)

    return results

def generateProtein(coords, chromname, files, resou, width=11, num_workers=36):
    manager = Manager()
    progress_queue = manager.Queue()

    coords = np.array(coords)
    valid_x, valid_y = coords[:, 0], coords[:, 1]

    chunk_size = len(valid_x) // num_workers
    chunks = [list(zip(valid_x, valid_y))[i:i + chunk_size] for i in range(0, len(valid_x), chunk_size)]
    params = [(chunk, chromname, files, resou, width, progress_queue) for chunk in chunks]

    total_tasks = len(valid_x)
    pbar = tqdm(total=total_tasks, desc='Processing Protein Data', leave=True)

    def update_progress():
        while True:
            item = progress_queue.get()
            if item is None:
                break
            pbar.update(1)

    # Start the progress updater thread
    progress_thread = Thread(target=update_progress)
    progress_thread.start()

    final_results = []
    with Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(generateProtein_worker, params):
            for matrix in result:
                if matrix.shape == (23, 23):
                    final_results.append(matrix)
        progress_queue.put(None)

    progress_thread.join()
    pbar.close()

    return final_results



def bilateral(data, gauss_scaling=100, bilat_scaling=10, epsilon=1e-3):
    """
    Apply bilateral filtering to one-dimensional data.

    Parameters:
    data (np.ndarray): One-dimensional input data array.
    gauss_scaling (float): Scaling factor for Gaussian width. Default is 100.
    bilat_scaling (float): Scaling factor for bilateral filter width. Default is 10.
    epsilon (float): Small value added for numerical stability. Default is 1e-3.

    Returns:
    np.ndarray: Bilaterally filtered data.
    """

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

    return bilateral_data


def calculate_expected(M, maxdis, raw=False):
    n = M.shape[0]
    R, C = M.nonzero()
    valid_pixels = np.isfinite(M.data)
    # extract valid columns
    if raw:
        R, C, data = R[valid_pixels], C[valid_pixels], M.data[valid_pixels]
        M = csr_matrix((data, (R, C)), shape=M.shape, dtype=float)
        marg = np.array(M.sum(axis=0)).ravel()
        valid_cols = marg > 0
    else:
        R, C = set(R[valid_pixels]), set(C[valid_pixels])
        valid_cols = np.zeros(n, dtype=bool)
        for i in R:
            valid_cols[i] = True
        for i in C:
            valid_cols[i] = True

    # calculate the expected value for each genomic distance
    exp_arr = np.zeros(maxdis + 1)
    for i in range(maxdis + 1):
        if i == 0:
            valid = valid_cols
        else:
            valid = valid_cols[:-i] * valid_cols[i:]

        diag = M.diagonal(i)
        diag = diag[valid]
        if diag.size > 10:
            exp = diag.mean()
            exp_arr[i] = exp

    # make exp_arr stringently non-increasing
    IR = IsotonicRegression(increasing=False, out_of_bounds='clip')
    _d = np.where(exp_arr > 0)[0]
    IR.fit(_d, exp_arr[_d])
    exp_arr = IR.predict(list(range(maxdis + 1)))

    return exp_arr


def distance_normaize_core(sub, exp_bychrom, x, y, w):

    # calculate x and y indices
    x_arr = np.arange(x - w, x + w + 1).reshape((2 * w + 1, 1))
    y_arr = np.arange(y - w, y + w + 1)

    D = y_arr - x_arr
    D = np.abs(D)
    min_dis = D.min()
    max_dis = D.max()
    if max_dis >= exp_bychrom.size:
        return sub
    else:
        exp_sub = np.zeros(sub.shape)
        for d in range(min_dis,
                       max_dis + 1):
            xi, yi = np.where(D == d)
            for i, j in zip(xi, yi):
                exp_sub[i, j] = exp_bychrom[d]

        normed = sub / exp_sub

        return normed
