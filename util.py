import numpy as np
import subprocess
import logging
import os
import re
import torch
import random
import datetime
import pickle, json
from functools import reduce
from scipy.stats import rankdata

def checkPath(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return

def cal_ranks(scores, labels, filters):
    target_ids = np.nonzero(labels)
    target_preds = scores[target_ids]
    target_batchs = target_ids[0]
    ranks = []
    for i in range(len(target_preds)):
        pred = scores[target_batchs[i]]
        mask = 1-filters[target_batchs[i]]
        pos_pred = target_preds[i]
        rank = np.sum((pred >= pos_pred) * mask) + 1
        ranks.append(rank)
    return ranks

def cal_performance(ranks):
    mrr = (1. / ranks).sum() / len(ranks)
    h_1 = sum(ranks<=1) * 1.0 / len(ranks)
    h_3 = sum(ranks <= 3) * 1.0 / len(ranks)
    h_10 = sum(ranks<=10) * 1.0 / len(ranks)
    return mrr, h_1, h_3, h_10

def cal_mrr_per_rel(ranks, rels, n_rel):
    rel_mrr = []
    for rel_id in range(n_rel * 2 + 1):
        rel_mask = rels == rel_id
        mrr = (1. / ranks * rel_mask).sum() / max(1, rel_mask.sum())
        rel_mrr.append(mrr)
    return np.array(rel_mrr)


def select_gpu():
    nvidia_info = subprocess.run('nvidia-smi', stdout=subprocess.PIPE)
    gpu_info = False
    gpu_info_line = 0
    proc_info = False
    gpu_mem = []
    gpu_occupied = set()
    i = 0
    for line in nvidia_info.stdout.split(b'\n'):
        line = line.decode().strip()
        if gpu_info:
            gpu_info_line += 1
            if line == '':
                gpu_info = False
                continue
            if gpu_info_line % 3 == 2:
                mem_info = line.split('|')[2]
                used_mem_mb = int(mem_info.strip().split()[0][:-3])
                gpu_mem.append(used_mem_mb)
        if proc_info:
            if line == '|  No running processes found                                                 |':
                continue
            if line == '+-----------------------------------------------------------------------------+':
                proc_info = False
                continue
            proc_gpu = int(line.split()[1])
            #proc_type = line.split()[3]
            gpu_occupied.add(proc_gpu)
        if line == '|===============================+======================+======================|':
            gpu_info = True
        if line == '|=============================================================================|':
            proc_info = True
        i += 1
    for i in range(0,len(gpu_mem)):
        if i not in gpu_occupied:
            logging.info('Automatically selected GPU %d because it is vacant.', i)
            return i
    for i in range(0,len(gpu_mem)):
        if gpu_mem[i] == min(gpu_mem):
            logging.info('All GPUs are occupied. Automatically selected GPU %d because it has the most free memory.', i)
            return i

def uniqueWithoutSort(a):
    indexes = np.unique(a, return_index=True)[1]
    res = [a[index] for index in sorted(indexes)]
    return res

def edge_match(edge_index, query_index):
    # O((n + q)logn) time
    # O(n) memory
    # edge_index: big underlying graph
    # query_index: edges to match

    # preparing unique hashing of edges, base: (max_node, max_relation) + 1
    base = edge_index.max(dim=1)[0] + 1
    # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
    # idea: max number of edges = num_nodes * num_relations
    # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
    # given a tuple (h, r), we will search for all other existing edges starting from head h
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    # scale2 = scale[-1] // scale
    scale = torch.div(scale[-1], scale, rounding_mode='trunc')

    # hash both the original edge index and the query index to unique integers
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

    # matched ranges: [start[i], end[i])
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
    num_match = end - start

    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)

    return order[range], num_match

def index_to_mask(index, size):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask

def masked_softmax(x, mask, **kwargs):
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)

def myprint(text, logger):
    if logger is not None:
        logger.info(text)
    else:
        print(text)

def logConfig(config, logger):
    d = config.__dict__
    for var in d:
        p = re.compile("__.*__")
        m = p.search(var)
        if m == None:
            myprint("config.%s=%s" % (var, d[var]), logger)

def init_logger(config):
    logger = logging.getLogger(config.startTimeSpan)
    logger.setLevel(level=logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logname = config.save_path + config.save_file_name +"/log.txt"
    handler = logging.FileHandler(logname)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(ch)
    logConfig(config, logger)
    return logger

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    import torch
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速

def save_pickle(data, file_name):
    f = open(file_name, "wb")
    pickle.dump(data, f)
    f.close()

def load_pickle(file_name):
    f = open(file_name, "rb")
    data = pickle.load(f)
    f.close()
    return data

def save_json(data, file_name):
    with open(file_name, "w") as f:
        json.dump(data, f)

def load_json(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data

# data segment
def get_batch_num(data_size, batch_size):
    return data_size // batch_size + (data_size % batch_size > 0)

def get_mmapped_features(features_path,features_shape,features_dtype):
    features = np.memmap(features_path, mode='r', shape=tuple(features_shape), dtype=features_dtype)
    return features

def to_list(arr):
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    elif isinstance(arr, list):
        return arr
    else:
        return [arr]

def to_array(arr):
    if isinstance(arr, np.ndarray):
        return arr
    elif isinstance(arr, list):
        return np.array(arr)
    else:
        return np.array([arr])