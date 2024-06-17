import numpy as np
from util import get_batch_num

class Compresser(object):
    def __init__(self, slice_size, entity_num, relation_num):
        self.slice_size = slice_size
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.uint32_size = np.iinfo(np.uint32).max #4294967295
        self.uint64_size = np.iinfo(np.uint64).max #18446744073709551615
        self.int32_size = np.iinfo(np.int32).max 
        self.int64_size = np.iinfo(np.int64).max 
        self.selectDataType()

    def selectDataType(self):
        left_part = self.entity_num
        right_part = max(self.slice_size, self.relation_num * 2)  # two types:  ent + len(slice_size), ent + rel
        right_len = len(str(right_part)) + 1 # add a mask position
        max_value = left_part * (10**right_len) + right_part * 10 + 1
        self.right_len = right_len
        if max_value < self.int32_size:
            self.datatype = np.int32
        elif max_value < self.int64_size:
            self.datatype = np.int64
        else:
            self.datatype = None


    def encode(self, triple_list): # one slice contains multiple atoms
        slice_vec = np.zeros(self.slice_size, dtype=self.datatype)
        pointer = 0
        for triple_part in triple_list:
            if len(triple_part) == 0: continue
            triple_part = triple_part.astype(self.datatype)
            part_len = len(triple_part)
            head_ent = triple_part[0][0]
            assert (triple_part[:,0] - head_ent).sum()==0 # 验证这一组三元组头实体相同
            # 存储首行，头实体和三元组数量，末尾标志位1
            head_value = head_ent * (10**self.right_len) + part_len * 10 + 1 # ent, len, 1
            slice_vec[pointer] = head_value
            # 存储三元组的r，t pairs
            item_vecs = triple_part[:,2] * (10**self.right_len) + triple_part[:,1] * 10 + 2 # ent, rel, 0
            slice_vec[pointer+1: pointer+len(item_vecs)+1] = item_vecs
            pointer += len(item_vecs)+1
        return slice_vec

    def encode2multi(self, triple_list): # one atom requires multiple slices
        head_ent = triple_list[0][0].astype(self.datatype) # 注意此处的类型声明不能去掉
        assert (triple_list[:, 0] - head_ent).sum() == 0  # 验证这一组三元组头实体相同
        batch_size = self.slice_size-1
        batch_num = get_batch_num(len(triple_list), batch_size)
        slice_list = []
        for bid in range(batch_num):
            slice_vec = np.zeros(self.slice_size, dtype=self.datatype)
            start = bid * batch_size
            end = min((bid+1)*batch_size, len(triple_list))
            triple_part = triple_list[start:end]
            if len(triple_part) == 0: break
            part_len = len(triple_part)
            # 存储首行，头实体和三元组数量，末尾标志位1
            head_value = head_ent * (10 ** self.right_len) + part_len * 10 + 1  # ent, len, 2
            slice_vec[0] = head_value
            # 存储三元组的r，t pairs
            triple_part = triple_part.astype(self.datatype)
            item_vecs = triple_part[:, 2] * (10 ** self.right_len) + triple_part[:, 1] * 10 + 2  # ent, rel, 0
            slice_vec[1: len(item_vecs) + 1] = item_vecs
            slice_list.append(slice_vec)
        return slice_list

    def decode(self, slice_vec):
        def is_end_with_one(n): return n % 10 == 1
        head_mask = np.vectorize(is_end_with_one)(slice_vec)
        head_index = np.nonzero(head_mask)[0] # 首行索引
        slice_vec = slice_vec // 10 # 排除最后标志位
        left_vec = (slice_vec // (10**(self.right_len-1))).astype(np.int64) # 拆解
        right_vec = (slice_vec % (10**(self.right_len-1))).astype(np.int64)
        triple_list = []
        for index in head_index:
            part_len = right_vec[index] # triple num
            head_ent = left_vec[index].repeat(part_len)
            tail_ent = left_vec[index+1: index+part_len+1]
            rel = right_vec[index+1: index+part_len+1]
            assert head_ent.shape == tail_ent.shape
            triples = np.stack([head_ent, rel, tail_ent, np.ones(len(head_ent))], axis=1)
            triple_list.append(triples)
        total_triples = np.concatenate(triple_list, axis=0)
        return total_triples

    def multi_decode(self, slice_vecs):
        # print(slice_vecs.shape) # n * 2048
        flatten_vec = slice_vecs.flatten()
        flatten_vec = flatten_vec[flatten_vec!=0] # 排除全0项
        head_mask = flatten_vec & 0x1 # bitwise_and判断最后一位是否为1
        head_index = np.nonzero(head_mask)[0]  # 首行索引是有用的
        head_vecs = flatten_vec[head_index] // 10
        head_vec = (head_vecs // (10 ** (self.right_len - 1))).astype(np.int64)  # 拆解
        len_vec = (head_vecs % (10 ** (self.right_len - 1))).astype(np.int64)
        # 筛选不重复的atoms，并保留大点的多个切片
        head_set = set()
        head_mask_vec = np.zeros_like(flatten_vec) # 记录目标flatten位置行的head值
        for i, index in enumerate(head_index):
            part_len = len_vec[i]  # triple num
            line = head_vecs[i].item()
            # print(i, line, head_set)
            if line in head_set and part_len != self.slice_size-1: # 排除重复点，同时保留大点切片
                continue
            else:
                head_set.add(line)
                head_mask_vec[index+1:index+part_len+1] = head_vec[i] + 1 # 此处加1防止ent=0
        # 得到的head_mask_vec和flatten_vec等长，三元组索引对应的值为其head值+1，同时作为mask和head存储
        # 根据mask批量处理所有三元组，避免循环累加耗时
        line_mask = head_mask_vec != 0
        final_head_ent = head_mask_vec[line_mask] - 1 # 头实体向量需要-1
        final_lines = flatten_vec[line_mask] // 10
        final_tail_ent = (final_lines // (10 ** (self.right_len - 1))).astype(np.int64)  # 拆解
        final_rel = (final_lines % (10 ** (self.right_len - 1))).astype(np.int64)
        total_triples = np.stack([final_head_ent, final_rel, final_tail_ent, np.ones(len(final_head_ent))], axis=1)
        return total_triples


