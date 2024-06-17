from util import *
from collections import Counter

class SamplerModule(object):
    def __init__(self, args):
        self.args = args

    def eval_unsampling(self, loader, batch_size, superbatch_size, n_data_num, sb_idx, mode):
        sbatch_input_triples, sbatch_label_csrmat = [], []  # 记录每个batch的数据
        start_nodes = Counter()
        for b_idx in range(superbatch_size):
            start = (sb_idx * superbatch_size + b_idx) * batch_size
            end = min(n_data_num, start + batch_size)
            if end <= start: break  # 到达训练集末尾
            triple_idx = np.arange(start, end)
            triple_batch, label_csrmat = loader.get_batch(triple_idx, data=mode, remove_big_node=False)  # 获取batch数据
            sbatch_input_triples.append(triple_batch)
            start_nodes.update(triple_batch[:, 0].tolist())
            sbatch_label_csrmat.append(label_csrmat)
        start_nodes = sorted(start_nodes.items(), key=lambda x: x[1], reverse=True)
        start_nodes = np.array(start_nodes)  # 从大到小排序<eid, count>
        return sbatch_input_triples, sbatch_label_csrmat, start_nodes

    def basic_sampling(self, loader, batch_size, superbatch_size, n_data_num, sb_idx, mode):
        sbatch_input_triples, sbatch_label_csrmat = [], []  # 记录每个batch的数据
        start_nodes = Counter()
        for b_idx in range(superbatch_size):
            start = (sb_idx * superbatch_size + b_idx) * batch_size
            end = min(n_data_num, start + batch_size)
            if end <= start: break  # 到达训练集末尾
            triple_idx = np.arange(start, end)
            triple_batch, label_csrmat = loader.get_batch(triple_idx, data=mode, remove_big_node=False)  # 获取batch数据
            sbatch_input_triples.append(triple_batch)
            start_nodes.update(triple_batch[:, 0].tolist())
        start_nodes = sorted(start_nodes.items(), key=lambda x: x[1], reverse=True)
        start_nodes = np.array(start_nodes)
        return sbatch_input_triples, start_nodes
