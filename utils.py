# coding=utf-8

# Read information from the hic header

import struct
import io
import os
import hicstraw


def csr_contact_matrix(norm, hicfile, chr1loc, chr2loc, unit,
                       binsize, is_synapse=False):
    '''
    Extract the contact matrix from .hic in CSR sparse format
    '''

    from scipy.sparse import csr_matrix
    import numpy as np
    import hicstraw

    tri_list = hicstraw.straw('observed',norm, hicfile, chr1loc,
                           chr2loc, unit, binsize)
    # for i in range(len(tri_list)):
    #     print("{0}\t{1}\t{2}".format(tri_list[i].binX, tri_list[i].binY, tri_list[i].counts))

    row = [r.binX//binsize for r in tri_list]
    col = [c.binY//binsize for c in tri_list]
    value = [item.counts for item in tri_list]
    N = max(col) + 1

    # re-scale KR matrix to ICE-matrix range
    M = csr_matrix((value, (row, col)), shape=(N, N))
    margs = np.array(M.sum(axis=0)).ravel() + \
        np.array(M.sum(axis=1)).ravel() - M.diagonal(0)
    margs[np.isnan(margs)] = 0
    scale = margs[margs != 0].mean()
    row, col = M.nonzero()
    value = M.data / scale
    M = csr_matrix((value, (row, col)), shape=(N, N))

    return M


def get_hic_chromosomes(hicfile, res):

    hic_info = read_hic_header(hicfile)
    chromosomes = []
    # handle with inconsistency between .hic header and matrix data
    for c, Len in hic_info['chromsizes'].items():

        if c == 'All' or c == 'MT':
            continue
        try:
            loc = '{0}:{1}:{2}'.format(c, 0, min(Len, 100000))
            # print(f"Trying to read chromosome {c} with location {loc}")
            _ = hicstraw.straw('observed','NONE', hicfile, loc, loc, 'BP', res)
            chromosomes.append(c)
        except:
            pass

    return chromosomes


def find_chrom_pre(chromlabels):

    ini = chromlabels[0]
    if ini.startswith('chr'):
        return 'chr'

    else:
        return ''


def readcstr(f):
    buf = ""
    while True:
        b = f.read(1)
        b = b.decode('utf-8', 'backslashreplace')

        if b is None or b == '\0':
            return str(buf)
        else:
            buf = buf + b


def read_hic_header(hicfile):

    if not os.path.exists(hicfile):
        # print(" buchunzai")
        return None  # probably a cool URI

    req = open(hicfile, 'rb')
    # print(req.read(3))
    magic_string = struct.unpack('<3s', req.read(3))[0]
    # print(magic_string)
    req.read(1)     #再读1个字节,这个字节是版本号
    if (magic_string != b"HIC"):
        # print("bushiyouxiaowenjian")
        return None  # this is not a valid .hic file


    info = {}
    version = struct.unpack('<i', req.read(4))[0]
    info['version'] = str(version)

    masterindex = struct.unpack('<q', req.read(8))[0]
    info['Master index'] = str(masterindex)


    genome = ""
    c = req.read(1).decode("utf-8")
    while (c != '\0'):
        genome += c
        c = req.read(1).decode("utf-8")
    info['Genome ID'] = str(genome)


    nattributes = struct.unpack('<i', req.read(4))[0]
    attrs = {}
    for i in range(nattributes):
        key = readcstr(req)
        value = readcstr(req)
        attrs[key] = value
    info['Attributes'] = attrs

    nChrs = struct.unpack('<i', req.read(4))[0]
    chromsizes = {}
    for i in range(nChrs):
        name = readcstr(req)
        length = struct.unpack('<i', req.read(4))[0]
        if name != 'ALL':
            chromsizes[name] = length
    info['chromsizes'] = chromsizes

    #以基因组坐标(碱基对)为单位的多个分辨率信息,并保存到了info字典中。
    info['Base pair-delimited resolutions'] = []
    nBpRes = struct.unpack('<i', req.read(4))[0]
    for i in range(nBpRes):
        res = struct.unpack('<i', req.read(4))[0]
        info['Base pair-delimited resolutions'].append(res)


    info['Fragment-delimited resolutions'] = []
    nFrag = struct.unpack('<i', req.read(4))[0]
    for i in range(nFrag):
        res = struct.unpack('<i', req.read(4))[0]
        info['Fragment-delimited resolutions'].append(res)

    return info
