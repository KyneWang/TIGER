from tqdm import tqdm
from util import *
from collections import Counter

class SlicerModule(object):
    def __init__(self, args, max_slice_num = 10):
        self.args = args
        self.match_mode = args.match_mode
        self.generate_mode = args.generate_mode
        self.n_layer = args.n_layer
        self.slice_size = args.slice_size
        self.item_size = 4
        self.max_slice_num = max_slice_num
        self.slice_path = os.path.join(args.save_path + args.save_file_name, "slicedata")
        self.mmapped_slices = None
        if not os.path.exists(self.slice_path):  os.makedirs(self.slice_path)

    def initialize(self, loader, compressor):
        self.slice_file = os.path.join(self.slice_path, "slice_d" + str(self.slice_size) + "_L" + str(self.n_layer) + ".dat")
        if os.path.exists(self.slice_file):
            os.remove(self.slice_file)
            file = open(self.slice_file, "wb")
            file.close()
        self.ifexist = np.zeros(loader.n_ent)
        self.mapping_dict = {}
        self.slice2atom_list = []
        self.slice2atom_2vec = [[], []]
        self.bignode2slice_dict = {}
        self.total_slice_num = 0
        self.mapping_mat = None
        self.loader = loader
        self.compressor = compressor
        self.ifcompress = self.args.ifcompress
        self.slice2node_pointer = []
        self.node2slice_pointer = []
        for i in range(self.loader.n_ent):
            self.node2slice_pointer.append([])
        if self.ifcompress:
            self.bignode_preslice()
        else:
            self.bignode_preslice0()

    ######################## CoreFunction ################################
    def slice_nodes(self, node_data, mode = "train", showTime=False):
        nodes, counts = node_data[:, 0], node_data[:, 1]
        self.bignode_samplesize = self.loader.big_node_threshold // 10
        node_mask = self.loader.load_smallnodeMask(nodes, self.loader.big_node_threshold)
        nodes_small, nodes_big = nodes[node_mask], nodes[~node_mask]
        directNei_list = self.loader.load_neighborhood(nodes_big, 1, remove_big_node=False, with_counts=False)
        directNei_nodes = set()
        self.directNei_dict = {}
        for i, node in enumerate(nodes_big):
            single_mask = self.loader.load_smallnodeMask(directNei_list[i], self.loader.big_node_threshold)
            single_nei_list = np.array(directNei_list[i])[single_mask].tolist()
            random.shuffle(single_nei_list)
            sampled_ents = single_nei_list[:self.bignode_samplesize] #self.slice_size] self.loader.big_node_threshold//10 # 限制大点扩散的范围
            if node not in sampled_ents: sampled_ents.append(node) # 添加该点本身
            directNei_nodes.update(sampled_ents)
            self.directNei_dict[node] = sampled_ents
        for node in nodes_small:
            self.directNei_dict[node] = [node]
            directNei_nodes.add(node)
        directNei_nodes = np.array(list(directNei_nodes), dtype=np.int64)
        unsliced_nodes = directNei_nodes[self.ifexist[directNei_nodes] == 0]
        if len(unsliced_nodes) == 0:
            return 0, 0
        previous_slice_num = self.total_slice_num
        n_layer = self.n_layer - 1
        node_neighborhoods = self.loader.load_neighborhood(unsliced_nodes, n_layer=n_layer)
        pbar = tqdm(range(len(unsliced_nodes)), desc="SliceNodes")
        slice_set = set()
        slice_summary, cost_summary = [], []
        for i in pbar:
            slice_ids, (big,old,new) = self.slice_process(unsliced_nodes[i], node_neighborhoods[i],
                                                          self.match_mode, self.generate_mode)  # 单个结果的切片
            total_num = sum([self.loader.load_atomNums(nid, mode="e2t") for nid, _ in node_neighborhoods[i]])
            slice_set.update(slice_ids)
            slice_summary.append([len(slice_ids), max(1, total_num / self.slice_size), big, old, new])
        
        slice_summary_vec = np.array(slice_summary)
        redundancy_vec = slice_summary_vec[:, 0] / slice_summary_vec[:, 1]
        mean_redundancy_rate = redundancy_vec.mean()
        mean_reuse_rate = len(slice_set) / slice_summary_vec[:, 0].sum()
            
        del node_neighborhoods
        if self.ifcompress:
            self.constuct_slice_data(self.slice2atom_list[previous_slice_num:])
        else:
            self.constuct_slice_data0(self.slice2atom_list[previous_slice_num:])
        return mean_redundancy_rate, mean_reuse_rate

    def slice_process(self, node_id, nei_vec, match_mode = 1, generate_mode=2): # reused slice
        origin_nei_ids = nei_vec[:,0]
        # filter big node
        showTime = False
        if showTime: sub_time = datetime.datetime.now()
        match_mask, big_slice_ids = self.bignode_match(nei_vec)
        nei_vec = nei_vec[match_mask]
        if showTime: print("---Slice bignode costtime(s)", (datetime.datetime.now() - sub_time).total_seconds(),
                           len(big_slice_ids), match_mask.sum() / len(nei_vec))
        # match process
        if showTime: sub_time = datetime.datetime.now()
        match_mask, old_slice_ids = self.slice_matching(nei_vec, mode = match_mode)
        gen_nei_vec = nei_vec[match_mask]
        if showTime: print("---Slice matching costtime(s)", (datetime.datetime.now() - sub_time).total_seconds(),
                           len(old_slice_ids), match_mask.sum() / len(nei_vec))
        # slice process
        if showTime: sub_time = datetime.datetime.now()
        new_slice_ids = []
        nei_ids,nei_hops = [],[]
        if len(gen_nei_vec) > 0:
            nei_ids, nei_hops = gen_nei_vec[:,0], gen_nei_vec[:,1]
            nei_weights = self.loader.load_atomNums(nei_ids, mode="e2t")

            if generate_mode == 4:
                hops2_mask = nei_vec[:,1] == 2
                whole_hop2_neis = nei_vec[:,0][hops2_mask]
                whole_hop2_weights = self.loader.load_atomNums(whole_hop2_neis, mode="e2t")
                whole_hop2_neis = sorted(zip(whole_hop2_neis, whole_hop2_weights), key=lambda x: x[1], reverse=True)
                whole_hop2_neis = list(zip(whole_hop2_neis, whole_hop2_weights))
            else:
                whole_hop2_neis = None

            new_slice_ids = self.slice_generate(nei_ids, nei_hops, nei_weights, whole_hop2_neis, mode = generate_mode)
        if showTime: print("---Slice generate costtime(s)", (datetime.datetime.now() - sub_time).total_seconds(),
                           len(new_slice_ids), len(nei_ids) / len(gen_nei_vec))
        # combine slices
        total_slice_ids = old_slice_ids + new_slice_ids + big_slice_ids
        self.mapping_dict[node_id] = total_slice_ids  # slice list
        self.ifexist[node_id] = node_id in self.mapping_dict
        return total_slice_ids, (len(big_slice_ids), len(old_slice_ids), len(new_slice_ids))

    def get_slice_list(self, node_list):
        neiNode_list = set()
        for node_id in node_list:
            directnei_list = self.directNei_dict[node_id]
            neiNode_list.update(directnei_list)
        node_list = list(neiNode_list)
        slice_sid_list, node_address_list = [], []
        for node_id in node_list:
            if node_id not in self.mapping_dict:
                print("getting unsliced node " + str(node_id) + " ...")
                break
            slice_list = self.mapping_dict[node_id]
            node_address_list.append([len(slice_sid_list), len(slice_list)])
            slice_sid_list.extend(slice_list)
        slice_set, slice_newid_list = np.unique(slice_sid_list, return_inverse=True)
        if len(slice_sid_list) > 0: assert len(slice_set) >= max(slice_newid_list)
        n2s_1hot = np.zeros([len(node_list), len(slice_set)])
        for i in range(len(node_list)):
            start, size = node_address_list[i]
            slice_ids = slice_newid_list[start:start + size]
            n2s_1hot[i][slice_ids] = 1
        return slice_set, n2s_1hot

    ######################## ProcessData ################################
    def constuct_slice_data0(self, new_slices):
        # construct new slice
        if self.item_size is None: self.item_size = 4 # [h,r,t, val]
        slice_neis = [item for single_slice in new_slices for item in single_slice]
        slice_nei_set = set(slice_neis)
        node_triple_dict = {}
        for nei_id in slice_nei_set:
            node_triples = self.loader.load_atomTriples(nei_id)
            node_triple_dict[nei_id] = node_triples
        savefile = open(self.slice_file, 'ab')
        for i, single_atom_set in enumerate(new_slices):
            single_slice_data = []
            for nei_id in single_atom_set:
                node_triples = node_triple_dict[nei_id] # self.loader.load_atomTriples(nei_id) #
                single_slice_data.extend(node_triples)
            new_slice_data = np.zeros([1, self.slice_size, self.item_size], dtype=np.int64)

            new_slice_data[0][:len(single_slice_data), :3] = single_slice_data
            new_slice_data[0][:len(single_slice_data), -1] = 1
            new_slice_data = new_slice_data.reshape(1, self.slice_size * self.item_size)

            new_slice_data.tofile(savefile)
            del new_slice_data
        del node_triple_dict
        savefile.close()
        self.mmapped_slices = get_mmapped_features(self.slice_file,
                                                   features_shape=[self.total_slice_num, self.slice_size * self.item_size],
                                                   features_dtype=np.int64)

    def constuct_slice_data(self, new_slices):
        # construct new slice
        slice_neis = [item for single_slice in new_slices for item in single_slice]
        slice_nei_set = set(slice_neis)
        node_triple_dict = {}
        for nei_id in slice_nei_set:
            node_triples = self.loader.load_atomTriples(nei_id)
            node_triple_dict[nei_id] = np.array(node_triples)
        savefile = open(self.slice_file, 'ab')
        for i, single_atom_set in enumerate(new_slices):
            single_slice_data = []
            triple_num = 0
            for nei_id in single_atom_set:
                node_triples = node_triple_dict[nei_id]
                if len(node_triples) > 0:
                    single_slice_data.append(node_triples)
                    triple_num += len(node_triples)
            new_slice_data = self.compressor.encode(single_slice_data)
            assert (new_slice_data!=0).sum() == triple_num + len(single_slice_data)
            new_slice_data = new_slice_data.reshape(1, self.slice_size)
            new_slice_data.tofile(savefile)
            del new_slice_data
        del node_triple_dict
        savefile.close()
        self.mmapped_slices = get_mmapped_features(self.slice_file,
                                              features_shape=[self.total_slice_num, self.slice_size],
                                              features_dtype=self.compressor.datatype)

    def get_slicer_data(self):
        return self.mmapped_slices

    ######################## BigNode ################################
    def bignode_preslice0(self):
        big_nei_mask = (self.loader.load_atomNums(mode="e2t") >= self.slice_size)
        self.big_nei_list = np.nonzero(big_nei_mask)[0]
        for nei_id in self.big_nei_list:
            node_triples = self.loader.load_atomTriples(nei_id)
            node_triples = np.concatenate([np.array(node_triples), np.ones([len(node_triples),1])], axis=1)
            if self.item_size is None: self.item_size = node_triples.shape[1]
            n_triples = len(node_triples)
            n_slices = get_batch_num(n_triples, self.slice_size)
            pedding_size = n_slices * self.slice_size - n_triples
            extended_node_triples = np.concatenate([node_triples, np.zeros([pedding_size, node_triples.shape[1]])], axis=0)
            node_slices = extended_node_triples.reshape(n_slices, self.slice_size, self.item_size).\
                                                reshape(n_slices, self.slice_size * self.item_size)
            node_slices = node_slices.astype(np.int64)
            slice2atomset = [[nei_id]] * len(node_slices)
            slice2atomset2 = [([nei_id], 1)] * len(node_slices)
            with open(self.slice_file, 'ab') as f:
                node_slices.tofile(f)
            self.slice2node_pointer.extend(slice2atomset2)
            self.node2slice_pointer[nei_id] = np.arange(self.total_slice_num, self.total_slice_num + len(node_slices)).tolist()
            self.slice2atom_list.extend(slice2atomset)
            for i, single_slice in enumerate(slice2atomset):
                self.slice2atom_2vec[0].extend([i + self.total_slice_num] * len(single_slice))
                self.slice2atom_2vec[1].extend(single_slice)
            self.total_slice_num = self.total_slice_num + len(node_slices)

    def bignode_preslice(self):
        big_nei_mask = (self.loader.load_atomNums(mode="e2t") >= self.slice_size)
        self.big_nei_list = np.nonzero(big_nei_mask)[0]
        for nei_id in self.big_nei_list:
            node_triples = self.loader.load_atomTriples(nei_id)
            slice_list = self.compressor.encode2multi(np.array(node_triples))
            node_slices = np.stack(slice_list, axis=0)
            slice2atomset = [[nei_id]] * len(node_slices)
            slice2atomset2 = [([nei_id], 1)] * len(node_slices)
            with open(self.slice_file, 'ab') as f:
                node_slices.tofile(f)
            self.slice2node_pointer.extend(slice2atomset2)
            self.node2slice_pointer[nei_id] = np.arange(self.total_slice_num, self.total_slice_num + len(node_slices)).tolist()
            self.slice2atom_list.extend(slice2atomset)
            for i, single_slice in enumerate(slice2atomset):
                self.slice2atom_2vec[0].extend([i + self.total_slice_num] * len(single_slice))
                self.slice2atom_2vec[1].extend(single_slice)
            self.total_slice_num = self.total_slice_num + len(node_slices)

    def bignode_match(self, nei_vec):
        nei_ids, nei_hops = nei_vec[:, 0], nei_vec[:, 1]
        nei2index_dict = {nid: id for id, nid in enumerate(nei_ids)}
        match_mask = np.ones(len(nei_ids))
        bignode_list = set(self.big_nei_list) & set(nei_ids)
        big_slice_ids = []
        if len(bignode_list) == 0:
            return match_mask.astype(bool), big_slice_ids
        for bignode_id in bignode_list:
            big_slice_ids.extend(self.node2slice_pointer[bignode_id])
            match_mask[nei2index_dict[bignode_id]] = 0
        return match_mask.astype(bool), big_slice_ids

    ######################## Matching ################################
    def slice_matching(self, nei_vec, mode=1):
        nei_ids, nei_hops = nei_vec[:, 0], nei_vec[:, 1]
        nei2index_dict = {nid: id for id, nid in enumerate(nei_ids)}
        slice_mask = np.ones(self.total_slice_num)
        uncovered_set = set(nei_ids)
        slice_id_list = []
        # This function check the reuse of node combination in completed slices
        nei_num = len(nei_ids)
        slice_scaner_flag = [1] * nei_num
        if nei_num == 0: return [], []

        past_slice_counter = Counter()
        nei_1hop_mask = (nei_hops == (nei_hops[0]-1))
        nei_1hop = nei_ids[nei_1hop_mask]
        sliced_nei_count = 0
        for i in range(len(nei_1hop)):
            if nei_1hop[i] in self.mapping_dict.keys():
                single_sliceids = self.mapping_dict[nei_1hop[i]]
                past_slice_counter.update(single_sliceids)
                sliced_nei_count += 1
        sorted_sliceids = sorted([item for item in past_slice_counter.keys() if past_slice_counter[item] > 1], key=lambda x: past_slice_counter[x], reverse=True)
        pointer = 0
        while (pointer < len(sorted_sliceids)):
            i = sorted_sliceids[pointer]
            if slice_mask[i] == 0:
                pointer += 1
                continue
            slice_mask[i] = 0
            triple_node_set, set_capacity = self.slice2node_pointer[i]
            if set_capacity >= self.args.min_capacity:
                if set(triple_node_set).issubset(uncovered_set):
                    slice_id_list.append(i)
                    uncovered_set = uncovered_set - triple_node_set
                    for il1 in triple_node_set:
                        slice_scaner_flag[nei2index_dict[il1]] = 0
            pointer += 1

        pointer = 0  # a pointer sliding in the neiid_list
        # when we still have candidates do not checked
        while (pointer < nei_num):
            # start_time = datetime.datetime.now()
            if slice_scaner_flag[pointer] == 0:
                pointer += 1
                continue
            # we check the slices of first node in remaining nodes
            temp_node = nei_ids[pointer]
            # the slice index that contain this node
            slice_of_node = self.node2slice_pointer[temp_node]
            sorted_nodesliceids = slice_of_node #sorted(slice_of_node, key=lambda x: self.slice2node_pointer[x][1], reverse=True)
            # means this node have not been loaded
            loaded0 = 0
            for i in sorted_nodesliceids: #[:3]
                if slice_mask[i] == 0: continue
                slice_mask[i] = 0
                # in i th slice, it contains these nodes
                triple_node_set, set_capacity = self.slice2node_pointer[i]
                # if all the nodes are in the remaining set, it can be used
                # if this slice can be reused, we update the candidate and scaner list
                if triple_node_set.issubset(uncovered_set):  # (len(triple_node_set-uncovered_set)==0):
                    loaded0 = 1
                    slice_id_list.append(i)
                    uncovered_set = uncovered_set - triple_node_set
                    for il1 in triple_node_set:
                        slice_scaner_flag[nei2index_dict[il1]] = 0
                    break
            # loaded0 == 0 means that this node have no slice can be reused, we pop the node in scaner for iteration,
            # so it will be preserved in waiting list for later generating new slices
            if (loaded0 == 0): pointer += 1
        # print("compute_times", compute_times, len(slice_id_list), len(uncovered_set))
        return np.array(slice_scaner_flag).astype(bool), slice_id_list

    ######################### Generate #################################
    def binary_search(self, id_list, weight_dict, new_id, start, end):
        if start > end:
            return start
        mid = (start + end) // 2
        if weight_dict[id_list[mid]][1] == weight_dict[new_id][1]:
            return mid
        elif weight_dict[id_list[mid]][1] < weight_dict[new_id][1]:
            return self.binary_search(id_list, weight_dict, new_id, start, mid - 1)
        else:
            return self.binary_search(id_list, weight_dict, new_id, mid + 1, end)

    def add_new_slice(self, current_slice, current_cap):
        self.slice2node_pointer.append([set(current_slice), current_cap / self.slice_size])
        completed_index_leng = len(self.slice2node_pointer) - 1
        if current_cap / self.slice_size >= self.args.min_capacity:
            for il3 in range(len(current_slice)):
                self.node2slice_pointer[current_slice[il3]].append(completed_index_leng)

    def slice_generate(self, nei_ids, nei_hops, nei_weights, whole_neis=None, mode=2):
        capacity = self.slice_size
        new_slices = []
        if sum(nei_weights)+len(nei_ids) <= self.slice_size:
            new_slices.append([nei_ids, sum(nei_weights)+len(nei_ids)])
        else:
            unsaved_nei_dict = {nei_ids[i]: nei_weights[i] + 1 for i in range(len(nei_ids))}
            if whole_neis is None:
                nei_mask = nei_hops == 2
                sorted_hopk_ids_weights = sorted(zip(nei_ids[nei_mask], nei_weights[nei_mask]), key=lambda x: x[1], reverse=True)
            else:
                sorted_hopk_ids_weights = whole_neis

            def ffd_packing(sorted_nei_id_weights, new_slices, unsaved_nei_dict):
                nei_ids, _ = zip(*sorted_nei_id_weights)
                old_slice_num = len(new_slices)
                for nei_id, weight in sorted_nei_id_weights:
                    flag = 0
                    for j, (slice, cap) in enumerate(new_slices):
                        if cap / capacity > self.args.min_capacity and j <= old_slice_num:
                            continue
                        if cap + weight <= capacity:
                            new_slices[j][0].append(nei_id)
                            new_slices[j][1] += weight
                            del unsaved_nei_dict[nei_id]
                            flag = 1
                            break
                    if flag == 0:
                        new_slices.append([[nei_id], weight])
                        del unsaved_nei_dict[nei_id]
                return new_slices, unsaved_nei_dict

            new_slices.append([[], 0])
            for cen_nid, cen_weight in sorted_hopk_ids_weights:
                if cen_nid not in unsaved_nei_dict: continue
                if unsaved_nei_dict is None or len(unsaved_nei_dict) == 0: break
                neis = self.loader.load_atomNeighbors(cen_nid)
                if cen_nid not in neis: neis.append(cen_nid)
                nei_id_weights = [[id, unsaved_nei_dict[id]] for id in neis if id in unsaved_nei_dict]
                sorted_nei_id_weights = nei_id_weights
                new_slices, unsaved_nei_dict = ffd_packing(sorted_nei_id_weights, new_slices, unsaved_nei_dict)
            if len(unsaved_nei_dict) > 0:
                new_slices2 = [[[],0]]
                input_ids_weights = list(unsaved_nei_dict.items())
                input_ids_weights = sorted(input_ids_weights, key=lambda x: x[1], reverse=True)
                new_slices2, unsaved_nei_dict = ffd_packing(input_ids_weights, new_slices2, unsaved_nei_dict)
                new_slices.extend(new_slices2)
        slice_count = 0
        for i, (current_slice, current_cap) in enumerate(new_slices):
            if len(current_slice) == 0: continue
            self.add_new_slice(current_slice, current_cap)
            self.slice2atom_list.append(current_slice)
            self.slice2atom_2vec[0].extend([i + self.total_slice_num] * len(current_slice))
            self.slice2atom_2vec[1].extend(current_slice)
            slice_count += 1
        new_slice_ids = np.arange(self.total_slice_num, self.total_slice_num + slice_count).tolist()
        self.total_slice_num = self.total_slice_num + slice_count
        return new_slice_ids



