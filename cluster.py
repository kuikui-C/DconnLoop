# coding=utf-8

from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy.signal import find_peaks, peak_widths
from sklearn.cluster import dbscan, MeanShift
from sklearn.cluster import OPTICS
from scipy.spatial.distance import euclidean
import warnings
import hdbscan
import argparse


def local_clustering(Donuts, min_count=3, r=2):
    final_list = []
    x = np.r_[[i[0] for i in Donuts]]
    y = np.r_[[i[1] for i in Donuts]]
    if x.size == 0:
        return final_list

    x_anchors = find_anchors(x, min_count=min_count, min_dis=r)
    y_anchors = find_anchors(y, min_count=min_count, min_dis=r)
    visited = set()
    lookup = set(zip(x, y))
    for x_a in x_anchors:
        for y_a in y_anchors:
            sort_list = []
            for i in range(x_a[1], x_a[2] + 1):
                for j in range(y_a[1], y_a[2] + 1):
                    if (i, j) in lookup:
                        sort_list.append((Donuts[(i, j)], (i, j)))
            sort_list.sort(reverse=True)
            _cluster_core(sort_list, r, visited, final_list)

    sort_list = []
    for i, j in zip(x, y):
        if (i, j) in visited:
            continue
        sort_list.append((Donuts[(i, j)], (i, j)))
    sort_list.sort(reverse=True)
    _cluster_core(sort_list, r, visited, final_list)

    x_summits = set([i[0] for i in x_anchors])
    y_summits = set([i[0] for i in y_anchors])
    for i, j in zip(x, y):
        if (i, j) in visited:
            continue

        if (i in x_summits) or (j in y_summits):
            final_list.append(((i, j), (i, j), 0))

    return final_list

def _cluster_core(sort_list, r, visited, final_list):
    # warnings.filterwarnings("ignore")
    pos = np.r_[[i[1] for i in sort_list]]
    if len(pos) >= 2:
        clusterer = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=3)
        clusterer.fit(pos)
        labels = clusterer.labels_

        pool = set()
        for i, p in enumerate(sort_list):
            if p[1] in pool:
                continue
            c = labels[i]
            if c == -1:
                continue
            sub = pos[labels == c]
            cen = p[1]
            rad = r
            Local = [p[1]]
            ini = -1
            while len(sub):
                out = []
                for q in sub:
                    if tuple(q) in pool:
                        continue
                    tmp = euclidean(q, cen)
                    if tmp <= rad:
                        Local.append(tuple(q))
                    else:
                        out.append(tuple(q))
                if len(out) == ini:
                    break
                ini = len(out)
                tmp = np.r_[Local]
                # assign centroid to a certain pixel
                cen = tuple(tmp.mean(axis=0).round().astype(int))
                rad = np.int64(np.round(max([euclidean(cen, q) for q in Local]))) + r
                sub = np.r_[out]
            for q in Local:
                pool.add(q)
            final_list.append((p[1], cen, rad))

        visited.update(pool)

def find_anchors(pos, min_count=3, min_dis=2, wlen=4):
    count = Counter(pos)
    refidx = range(min(count), max(count) + 1)
    signal = np.r_[[count[i] for i in refidx]]
    summits = find_peaks(signal, height=min_count, distance=min_dis)[0]
    sorted_summits = [(signal[i], i) for i in summits]
    sorted_summits.sort(reverse=True)

    peaks = set()
    records = {}
    for _, i in sorted_summits:
        tmp = peak_widths(signal, [i], rel_height=1, wlen=wlen)[2:4]
        li, ri = int(np.round(tmp[0][0])), int(np.round(tmp[1][0]))
        lb = refidx[li]
        rb = refidx[ri]
        if not len(peaks):
            peaks.add((refidx[i], lb, rb))
            for b in range(lb, rb + 1):
                records[b] = (refidx[i], lb, rb)
        else:
            for b in range(lb, rb + 1):
                if b in records:
                    # merge anchors
                    m_lb = min(lb, records[b][1])
                    m_rb = max(rb, records[b][2])
                    summit = records[b][0]
                    peaks.remove(records[b])
                    break
            else:
                m_lb, m_rb, summit = lb, rb, refidx[i]
            peaks.add((summit, m_lb, m_rb))
            for b in range(m_lb, m_rb + 1):
                records[b] = (summit, m_lb, m_rb)

    return peaks

def rhoDelta(data,resol,dc):
    pos = data[['s1', 's2']].to_numpy().astype(int) // resol
    val = data['raw_signal'].to_numpy().astype(float)

    posTree = KDTree(pos, leaf_size=30, metric='chebyshev')
    NNindexes, NNdists = posTree.query_radius(pos, r=dc, return_distance=True)

    # calculate local density rho
    rhos = []
    for i in range(len(NNindexes)):
        rhos.append(np.dot(np.exp(-(NNdists[i] / dc) ** 2), val[NNindexes[i]]))
    rhos = np.asarray(rhos)
    print(np.max(rhos), np.min(rhos), np.sum(rhos >= 2))

    # calculate delta_i, i.e. distance to nearest point with larger rho
    _r = 100
    _indexes, _dists = posTree.query_radius(pos, r=_r, return_distance=True, sort_results=True)
    deltas = rhos * 0
    LargerNei = rhos * 0 - 1
    for i in range(len(_indexes)):
        idx = np.argwhere(rhos[_indexes[i]] > rhos[_indexes[i][0]])
        if idx.shape[0] == 0:
            deltas[i] = _dists[i][-1] + 1
        else:
            LargerNei[i] = _indexes[i][idx[0]]
            deltas[i] = _dists[i][idx[0]]
    failed = np.argwhere(LargerNei == -1).flatten()
    while len(failed) > 1 and _r < 100000:
        _r = _r * 10
        _indexes, _dists = posTree.query_radius(pos[failed], r=_r, return_distance=True, sort_results=True)
        for i in range(len(_indexes)):
            idx = np.argwhere(rhos[_indexes[i]] > rhos[_indexes[i][0]])
            if idx.shape[0] == 0:
                deltas[failed[i]] = _dists[i][-1] + 1
            else:
                LargerNei[failed[i]] = _indexes[i][idx[0]]
                deltas[failed[i]] = _dists[i][idx[0]]
        failed = np.argwhere(LargerNei == -1).flatten()

    data['rhos']=rhos
    data['deltas']=deltas

    return data

def pool(dc,candidates,resol,minscore, rhosPR, deltasPR,output,refine):
    D = defaultdict(dict)
    score_pool = defaultdict(dict)
    with open(candidates, 'r') as source:
        for line in source:
            p = line.rstrip().split()
            c1, s1, s2, prob, v = p[0], int(p[1]), int(p[4]), float(p[6]), float(p[7])
            if prob >= minscore:
                D[c1][(s1 // resol, s2 // resol)] = v
                score_pool[c1][(s1 // resol, s2 // resol)] = [prob, v]

    data = []
    loopPds = []
    for c in D:
        tmp = local_clustering(D[c], min_count=3, r=2)
        for i in tmp:
            if i[0] in score_pool[c]:
                s1 = str(i[0][0] * resol)
                e1 = str(i[0][0] * resol + resol)
                s2 = str(i[0][1] * resol)
                e2 = str(i[0][1] * resol + resol)
                prob = str(score_pool[c][i[0]][0])
                raw_signal = str(score_pool[c][i[0]][1])
                data.append((c, s1, e1, c, s2, e2, prob, raw_signal))

        data_df = pd.DataFrame(data, columns=['chromosome', 's1', 'e1', 'chromosome2', 's2', 'e2', 'prob', 'raw_signal'])

        data_df[['rhos', 'deltas']] = 0
        data_df = rhoDelta(data_df, resol=resol, dc=dc).reset_index(drop=True)

        pos = data_df[['s1', 's2']].to_numpy().astype(int) // resol
        posTree = KDTree(pos, leaf_size=30, metric='chebyshev')


        rhos = data_df['rhos'].to_numpy()
        rhos_percentile = np.percentile(rhos, rhosPR, axis=None, out=None, overwrite_input=False, keepdims=False)

        deltas = data_df['deltas'].to_numpy()
        deltas_percentile = np.percentile(deltas, deltasPR, axis=None, out=None, overwrite_input=False, keepdims=False)

        centroid = np.argwhere((rhos > rhos_percentile) & (deltas > deltas_percentile)).flatten()

        _r = 100
        _indexes, _dists = posTree.query_radius(pos, r=_r, return_distance=True, sort_results=True)
        LargerNei = rhos * 0 - 1
        for i in range(len(_indexes)):
            idx = np.argwhere(rhos[_indexes[i]] > rhos[_indexes[i][0]])
            if idx.shape[0] == 0:
                pass
            else:
                LargerNei[i] = _indexes[i][idx[0]]

        failed = np.argwhere(LargerNei == -1).flatten()
        while len(failed) > 1 and _r < 100000:
            _r = _r * 10
            _indexes, _dists = posTree.query_radius(pos[failed], r=_r, return_distance=True, sort_results=True)
            for i in range(len(_indexes)):
                idx = np.argwhere(rhos[_indexes[i]] > rhos[_indexes[i][0]])
                if idx.shape[0] == 0:
                    pass
                else:
                    LargerNei[failed[i]] = _indexes[i][idx[0]]
            failed = np.argwhere(LargerNei == -1).flatten()

        # assign rest loci to loop clusters
        LargerNei = LargerNei.astype(int)
        label = LargerNei * 0 - 1
        for i in range(len(centroid)):
            label[centroid[i]] = i
        decreasingsortedIdxRhos = np.argsort(-rhos)
        for i in decreasingsortedIdxRhos:
            if label[i] == -1:
                label[i] = label[LargerNei[i]]

        # refine loop   (Select the point with the greatest probability as the refined loop point)
        val = data_df['prob'].to_numpy().astype(float)
        refinedLoop = []
        label = label.flatten()
        for l in set(label):
            idx = np.argwhere(label == l).flatten()
            if len(idx) > 0:
                refinedLoop.append(idx[np.argmax(val[idx])])

        if refine:
            loopPds.append(data_df.loc[refinedLoop])
        else:
            loopPds.append(data_df.loc[centroid])


    loopPd = pd.concat(loopPds).sort_values('prob', ascending=False)
    loopPd[['chromosome', 's1', 'e1', 'chromosome2', 's2', 'e2', 'prob', 'raw_signal']].to_csv(output, sep='\t', header=False, index=False)
    print(len(loopPd), 'loops saved to ', output)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run chromatin loop clustering and refinement.")

    parser.add_argument('-d', '--dc', type=int, default=5,
                        help='Distance cutoff for local density calculation.')
    parser.add_argument('-i', '--candidates', type=str, required=True,
                        help='Path to the candidates file.')
    parser.add_argument('-r', '--resol', type=int, default=10000,
                        help='Resolution in base pairs.')
    parser.add_argument('-m', '--minscore', type=float, default=0.97,
                        help='Minimum score for filtering candidates.')
    parser.add_argument('-p', '--rhosPR', type=int, default=75,
                        help='Percentile rank for rho selection.')
    parser.add_argument('-e', '--deltasPR', type=int, default=10,
                        help='Percentile rank for delta selection.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output file path.')
    parser.add_argument('-f', '--refine', action='store_true',
                        help='Flag to refine the loop clusters.')

    args = parser.parse_args()

    pool(dc=args.dc,
         candidates=args.candidates,
         resol=args.resol,
         minscore=args.minscore,
         rhosPR=args.rhosPR,
         deltasPR=args.deltasPR,
         output=args.output,
         refine=args.refine)

