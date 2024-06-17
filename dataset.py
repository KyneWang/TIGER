from tqdm import tqdm
from util import *
import datetime
import copy
import warnings
from scipy.sparse import csr_matrix
from collections import defaultdict, Counter

class DataLoader:
    def __init__(self, task_dir, big_node_threshold = 2000, feature_cache_size = 2e8, relation2id=None, mode="transductive"):
        self.task_dir = task_dir
        self.mode = mode
        self.relation2id = relation2id
        ### read data
        if mode=="transductive":
            self.predata_path = os.path.join(self.task_dir, "preprocess")
        elif mode == "inductive":
            self.predata_path = os.path.join(self.task_dir, "ind_preprocess")
        elif mode == "allinductive":
            self.predata_path = os.path.join(self.task_dir, "all_preprocess")
        if not os.path.exists(self.predata_path): os.makedirs(self.predata_path)
        if not os.path.exists(os.path.join(self.predata_path, "info.json")):
            self.read_data_process(showTime=True)
        self.dataset_info = load_json(os.path.join(self.predata_path, "info.json"))  # 加载预计算信息

        self.n_ent = self.dataset_info["n_ent"]
        self.n_rel = self.dataset_info["n_rel"]
        self.valid_q, self.valid_a = None, None
        self.test_q, self.test_a = None, None
        self.filters, self.trainfilters = None, None

        self.big_node_threshold = int(big_node_threshold)
        self.feature_cache_size = feature_cache_size
        self.neiCache = NeighborLoader(self.predata_path, self.dataset_info, feature_cache_size=self.feature_cache_size,
                                       mode=self.mode, big_node_threshold = self.big_node_threshold)
        self.neiCache.initialize()
        # self.neiCache.subgraph_generate(n_layer=2)

    def load_before_train(self, showTime=True):
        save_path = self.predata_path
        mode_name = "ind_" if self.mode == "inductive" else ""
        if showTime: start = datetime.datetime.now()
        self.train_data = load_pickle(os.path.join(save_path, mode_name+"train.pkl"))
        self.n_train = len(self.train_data)
        assert self.n_train == int(self.dataset_info["n_"+mode_name+"train"])
        if showTime: print("costtime(s)", (datetime.datetime.now() - start).total_seconds(), "load train set:", self.n_train)

        self.all_observed_data = self.train_data
        self.fact_data = load_pickle(os.path.join(save_path, mode_name+"facts.pkl"))
        self.n_facts = len(self.fact_data)
        assert self.n_facts == int(self.dataset_info["n_"+mode_name+"facts"])
        self.all_observed_data = np.concatenate(
            [self.fact_data[:self.n_facts // 2], self.train_data[:self.n_train // 2],
             self.fact_data[self.n_facts // 2:], self.train_data[self.n_train // 2:]], axis=0)
        if showTime: print("costtime(s)", (datetime.datetime.now() - start).total_seconds(), "load facts set:", self.n_facts)
        self.n_total_train = len(self.all_observed_data)
        self.trainfilters = load_pickle(os.path.join(save_path, "filter_train.pkl"))

    def load_before_evaluate(self, showTime=True):
        save_path = self.predata_path
        if showTime: start = datetime.datetime.now()
        if self.valid_q is None and self.valid_a is None:
            self.valid_q, self.valid_a = load_pickle(os.path.join(save_path, "valid.pkl"))
            self.n_valid = len(self.valid_q)
        assert self.n_valid == int(self.dataset_info["n_valid"])
        if self.test_q is None and self.test_a is None:
            self.test_q, self.test_a = load_pickle(os.path.join(save_path, "test.pkl"))
            self.n_test = len(self.test_q)
        assert self.n_test == int(self.dataset_info["n_test"])
        if self.filters is None:
            self.filters = load_pickle(os.path.join(save_path, "filter_test.pkl"))
            assert len(self.filters) == int(self.dataset_info["n_filters"])
        if showTime: print("costtime(s)", (datetime.datetime.now() - start).total_seconds(), "load evaluate data:",
                           self.n_valid, self.n_test)

    def read_data_process(self, showTime=False, ifRemoveRel=True):
        """
        :param showTime: whether print the running time and results
        :param ifRemoveRel: whether remove the sparse relations only existing in inductive test/valid triples
        :return: preprocessed KG data
        """
        dataset_info = {"path": self.task_dir}
        save_path = self.predata_path
        self.filters = defaultdict(lambda: set())
        self.trainfilters = defaultdict(lambda: set())
        self.indtrainfilters = defaultdict(lambda: set())
        train_rel_set = None # only useful when ifRemoveRel

        ### 1. read entities.txt and relations.txt
        if showTime: start = datetime.datetime.now()
        self.read_items()
        dataset_info["n_ent"] = self.n_ent
        dataset_info["n_rel"] = self.n_rel
        if showTime: print("costtime(s)", (datetime.datetime.now() - start).total_seconds(), end=" ")
        print('n_ent:', dataset_info["n_ent"], 'n_rel:', dataset_info["n_rel"])

        ### 2. read triples
        # train.txt
        if showTime: start = datetime.datetime.now()
        train_triple, train_triple_inv = self.read_triples("train.txt", mode="train")
        train_data = np.array(train_triple + train_triple_inv)
        dataset_info["n_train"] = len(train_data)
        save_pickle(train_data, os.path.join(save_path, "train.pkl")) # save
        del train_triple, train_triple_inv, train_data
        if showTime: print("costtime(s)", (datetime.datetime.now() - start).total_seconds(), 'n_train:', dataset_info["n_train"])

        # facts.txt
        if showTime: start = datetime.datetime.now()
        fact_triple, fact_triple_inv = self.read_triples("facts.txt", mode="train")
        fact_data = np.array(fact_triple + fact_triple_inv)
        dataset_info["n_facts"] = len(fact_data)
        save_pickle(fact_data, os.path.join(save_path, "facts.pkl")) # save
        del fact_triple, fact_triple_inv, fact_data
        if showTime: print("costtime(s)", (datetime.datetime.now() - start).total_seconds(), 'n_facts:', dataset_info["n_facts"])

        if self.mode != "transductive":
            # ind_train.txt
            if showTime: start = datetime.datetime.now()
            train_triple, train_triple_inv = self.read_triples("ind_train.txt", mode="ind_train")
            train_data = np.array(train_triple + train_triple_inv)

            if ifRemoveRel: train_rel_set = set(train_data[:, 1].tolist())

            dataset_info["n_ind_train"] = len(train_data)
            save_pickle(train_data, os.path.join(save_path, "ind_train.pkl"))  # save
            del train_triple, train_triple_inv, train_data
            if showTime: print("costtime(s)", (datetime.datetime.now() - start).total_seconds(), 'n_ind_train:',
                               dataset_info["n_ind_train"])
        if self.mode == "inductive":
            # ind_facts.txt
            if showTime: start = datetime.datetime.now()
            fact_triple, fact_triple_inv = self.read_triples("ind_facts.txt", mode="ind_train")
            fact_data = np.array(fact_triple + fact_triple_inv)
            dataset_info["n_ind_facts"] = len(fact_data)
            save_pickle(fact_data, os.path.join(save_path, "ind_facts.pkl"))  # save
            del fact_triple, fact_triple_inv, fact_data
            if showTime: print("costtime(s)", (datetime.datetime.now() - start).total_seconds(), 'n_ind_facts:', dataset_info["n_ind_facts"])
        if self.mode == "allinductive":
            # ind_facts.txt
            if showTime: start = datetime.datetime.now()
            fact_triple, fact_triple_inv = self.read_triples("all_facts.txt", mode="ind_train")
            fact_data = np.array(fact_triple + fact_triple_inv)
            dataset_info["n_ind_facts"] = len(fact_data)
            save_pickle(fact_data, os.path.join(save_path, "ind_facts.pkl"))  # save
            del fact_triple, fact_triple_inv, fact_data
            if showTime: print("costtime(s)", (datetime.datetime.now() - start).total_seconds(), 'n_ind_facts:', dataset_info["n_ind_facts"])

        # valid.txt
        if showTime: start = datetime.datetime.now()
        valid_triple, valid_triple_inv, valid_q, valid_a = self.read_triples('valid.txt', trainRelSet = train_rel_set, mode="test")
        dataset_info["n_valid"] = len(valid_q)
        save_pickle([valid_q, valid_a], os.path.join(save_path, "valid.pkl"))  # save
        del valid_triple, valid_triple_inv, valid_q, valid_a
        if showTime: print("costtime(s)", (datetime.datetime.now() - start).total_seconds(), 'n_valid:', dataset_info["n_valid"])
        # test.txt
        if showTime: start = datetime.datetime.now()
        test_triple, test_triple_inv, test_q, test_a = self.read_triples('test.txt', trainRelSet = train_rel_set, mode="test")
        dataset_info["n_test"] = len(test_q)
        save_pickle([test_q, test_a], os.path.join(save_path, "test.pkl")) # save
        del test_triple, test_triple_inv, test_q, test_a
        if showTime: print("costtime(s)", (datetime.datetime.now() - start).total_seconds(), 'n_test:', dataset_info["n_test"])

        ### 3. save data
        if showTime: start = datetime.datetime.now()
        for filt in self.filters: self.filters[filt] = list(self.filters[filt])
        for filt in self.trainfilters: self.trainfilters[filt] = list(self.trainfilters[filt])
        save_pickle(dict(self.filters), os.path.join(save_path, "filter_test.pkl"))
        save_pickle(dict(self.trainfilters), os.path.join(save_path, "filter_train.pkl"))
        dataset_info["n_filters"] = len(self.filters)
        dataset_info["n_trainfilters"] = len(self.trainfilters)
        if self.mode == "inductive":
            for filt in self.indtrainfilters: self.indtrainfilters[filt] = list(self.indtrainfilters[filt])
            save_pickle(dict(self.indtrainfilters), os.path.join(save_path, "filter_indtrain.pkl"))
            dataset_info["n_indtrainfilters"] = len(self.indtrainfilters)
        save_json(dataset_info, os.path.join(save_path, "info.json"))  # save  # json?
        del self.filters, self.trainfilters, self.indtrainfilters
        if showTime: print("costtime(s)", (datetime.datetime.now() - start).total_seconds(), 'save completed')

    def load_graph_process(self, showTime=False):
        return

    def read_process(self, showTime=False):
        ### 1. read entities.txt and relations.txt
        if showTime: start = datetime.datetime.now()
        self.read_items()
        if showTime: print("costtime(s)", (datetime.datetime.now()-start).total_seconds(), end=" ")
        print('n_ent:', self.n_ent, 'n_rel:', self.n_rel)

        ### 2. read triple sets
        if showTime: start = datetime.datetime.now()
        self.filters = defaultdict(lambda: set())
        self.trainfilters = defaultdict(lambda: set())
        train_triple, train_triple_inv = self.read_triples('train.txt', mode="train")
        valid_triple, valid_triple_inv, self.valid_q, self.valid_a = self.read_triples('valid.txt', mode="test")
        test_triple, test_triple_inv, self.test_q, self.test_a = self.read_triples('test.txt', mode="test")
        self.train_data = train_triple + train_triple_inv
        self.valid_data = valid_triple + valid_triple_inv
        self.test_data = test_triple + test_triple_inv
        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_q)
        self.n_test = len(self.test_q)
        if showTime: print("costtime(s)", (datetime.datetime.now() - start).total_seconds(), end=" ")
        print('n_train:', self.n_train, 'n_valid:', self.n_valid, 'n_test:', self.n_test)
        del train_triple, train_triple_inv, valid_triple, valid_triple_inv, test_triple, test_triple_inv

        ### 3. read fact sets
        if showTime: start = datetime.datetime.now()
        if self.mode == "transductive":
            fact_triple, fact_triple_inv = self.read_triples('facts.txt', mode="train")
            self.fact_data = fact_triple + fact_triple_inv
            self.all_observed_data = self.train_data + self.fact_data
            del fact_triple, fact_triple_inv
        else:
            self.all_observed_data = self.train_data
        self.train_data = np.array(self.train_data)
        self.total_train_data = np.array(self.all_observed_data)
        self.n_total_train = len(self.all_observed_data)
        if showTime: print("costtime(s)", (datetime.datetime.now() - start).total_seconds(), end=" ")
        print("all_observed_data", len(self.all_observed_data))

    def read_items(self):
        with open(os.path.join(self.task_dir, 'entities.txt'), encoding="utf8") as f:
            self.entity2id = dict()
            n_ent = 0
            for line in f:
                entity = line.strip().split()[0]
                self.entity2id[entity] = n_ent
                n_ent += 1

        if self.mode == "inductive":
            with open(os.path.join(self.task_dir, 'ind_entities.txt'), encoding="utf8") as f: # ind_entities
                for line in f:
                    entity = line.strip().split()[0]
                    self.entity2id[entity] = n_ent
                    n_ent += 1

        if self.mode == "allinductive":
            with open(os.path.join(self.task_dir, 'all_ents.txt'), encoding="utf8") as f: # ind_entities
                for line in f:
                    entity = line.strip().split()[0]
                    self.entity2id[entity] = n_ent
                    n_ent += 1

        if self.relation2id == None:
            with open(os.path.join(self.task_dir, 'relations.txt'), encoding="utf8") as f:
                self.relation2id = dict()
                n_rel = 0
                for line in f:
                    relation = line.strip().split()[0]
                    self.relation2id[relation] = n_rel
                    n_rel += 1
        else:
            n_rel = len(self.relation2id)
        self.n_ent = n_ent
        self.n_rel = n_rel

    def read_triples(self, filename, trainRelSet = None, mode="none"):
        triples, new_triples = [], []
        trip_hr = defaultdict(lambda: list())
        with open(os.path.join(self.task_dir, filename), encoding="utf8") as f:
            for line in f:
                items = line.strip().split()
                if len(items) != 3: continue
                h, r, t = items
                if "train" not in mode and trainRelSet is not None:
                    if int(r) not in trainRelSet: continue
                if r not in self.relation2id.keys(): continue
                if h not in self.entity2id.keys(): continue
                if t not in self.entity2id.keys(): continue
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                triples.append([h, r, t])
                new_triples.append([t, r + self.n_rel, h])
                if mode == "ind_train":
                    self.indtrainfilters[(h, r)].add(t)
                    self.indtrainfilters[(t, r + self.n_rel)].add(h)
                # load filters
                if mode == "train":
                    self.trainfilters[(h, r)].add(t)
                    self.trainfilters[(t, r + self.n_rel)].add(h)
                if mode != "ind_train":
                    self.filters[(h, r)].add(t)
                    self.filters[(t, r + self.n_rel)].add(h)
                # load quries
                if mode == "test":
                    trip_hr[(h, r)].append(t)
                    trip_hr[(t, r + self.n_rel)].append(h)
        if mode == "test":
            queries = []
            answers = []
            for i, key in enumerate(trip_hr): # build queries
                queries.append(key)
                answers.append(np.array(trip_hr[key]))
            return triples, new_triples, queries, answers
        else: # train
            return triples, new_triples

    def get_batch(self, batch_idx, data='train', remove_big_node = True, remove_reverse=False):
        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        if data == 'train':
            triples = np.array(self.train_data)[batch_idx]
            return triples, None
        if data == 'valid':
            query, answer = np.array(self.valid_q), self.valid_a
        if data == 'test':
            query, answer = np.array(self.test_q), self.test_a

        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
        objs = batch_idx

        if remove_reverse:
            reverse_mask = rels < self.n_rel
            subs = subs[reverse_mask]
            rels = rels[reverse_mask]
            objs = objs[reverse_mask]

        bid_list, obj_list = [], []
        for i in range(len(subs)):
            single_obj = answer[objs[i]]
            obj_list.extend(single_obj)
            bid_list.extend([i] * len(single_obj))
        assert len(obj_list) == len(bid_list)
        objs = csr_matrix((np.ones((len(bid_list),)), (np.array(bid_list), np.array(obj_list))),
                            shape=(len(batch_idx), self.n_ent))
        input_batch = np.stack([subs, rels, subs], axis=1)
        label_csrmat = objs
        return input_batch, label_csrmat

    def shuffle_train(self, ratio=0.25, remove_big_node = True, remove_one_loop=False, showTime=False):
        all_triple = np.array(self.all_observed_data[:self.n_total_train//2])
        random_mask = (np.random.random(len(all_triple)) <= ratio)
        train_triple = all_triple[random_mask]

        batch_triples = np.stack([train_triple[:, 2], train_triple[:, 1] + self.n_rel, train_triple[:, 0]], axis=1)
        train_data = np.concatenate([train_triple, batch_triples], axis=0)
        np.random.shuffle(train_data)
        self.train_data = train_data
        self.n_train = len(self.train_data)

    def load_entTriples(self, batch_nodes, node_hops = None, remove_big_node=True):
        return self.neiCache.load_entTriples(batch_nodes, node_hops, remove_big_node)
    def load_atomNeighbors(self, node_ids):
        return self.neiCache.load_atomNeighbors(node_ids)
    def load_atomTriples(self, node_id, remove_big_node = True, with_self_edge = True):
        return self.neiCache.load_atomTriples(node_id, remove_big_node, with_self_edge)
    def load_atomNums(self, node_ids=None, mode="e2t"):
        return self.neiCache.getAtomNums(node_ids, mode)
    def load_smallnodeMask(self, node_ids=None, threshold = 1000):
        return self.neiCache.getAtomNums(node_ids, mode="e2t") <= threshold
    def load_neighborhood(self, batch_nodes, n_layer = 3, remove_big_node = True, with_counts = True, verbose=True):
        return self.neiCache.load_neighborhood(batch_nodes, n_layer, remove_big_node, with_counts, verbose)
    def load_structured_neighborhood(self, batch_nodes, n_layer = 3, remove_big_node = True, verbose=True):
        return self.neiCache.load_structured_neighborhood(batch_nodes, n_layer, remove_big_node, verbose)
    def load_subgraph(self, nodes: object, n_layer: object = 3, remove_big_node: object = True, with_counts: object = False) -> object:
        return self.neiCache.load_subgraph(nodes, n_layer, remove_big_node, with_counts)
    def load_bignode_triples(self, nodes):
        return self.neiCache.load_bignode_triples(nodes)

class NeighborLoader(object):
    def __init__(self, data_path, data_info, feature_cache_size=2e8, mode="transductive", big_node_threshold = 2000):
        self.n_ent = data_info["n_ent"]
        self.n_rel = data_info["n_rel"]
        self.e2e_cache_size = int(feature_cache_size)
        self.e2t_cache_size = int(feature_cache_size)
        self.subG_cache_size = int(feature_cache_size)
        self.threshold = big_node_threshold
        self.save_path = data_path
        self.mode = mode
        self.readFullData = False

    def initialize(self):
        self.e2e_data, self.e2e_map, self.e2e_nums = self.data_initialize(mode = "e2e")
        self.e2t_data, self.e2t_map, self.e2t_nums = self.data_initialize(mode = "e2t")
        self.big_node_mask = self.e2t_nums > self.threshold
        self.cache_generate(sample_big_node=True, ifprint=False)

    def data_preprocess(self):
        save_path = self.save_path
        # load preprocess data from Loader
        dataset_info = load_json(os.path.join(save_path, "info.json"))
        self.n_ent = dataset_info["n_ent"]
        self.n_rel = dataset_info["n_rel"]
        trainfilters = load_pickle(os.path.join(save_path, "filter_train.pkl"))
        e2e_dict = {eid: set() for eid in range(self.n_ent)}
        e2t_dict = {eid: [] for eid in range(self.n_ent)}
        for filt in trainfilters:
            (eid, rid) = filt
            entlist = trainfilters[filt]
            e2e_dict[eid].update(entlist)
            triple_list = [(eid, rid, ent) for ent in entlist]
            e2t_dict[eid].extend(triple_list)

        if self.mode == "inductive":
            ind_trainfilters = load_pickle(os.path.join(save_path, "filter_indtrain.pkl"))
            for filt in ind_trainfilters:
                (eid, rid) = filt
                entlist = ind_trainfilters[filt]
                e2e_dict[eid].update(entlist)
                triple_list = [(eid, rid, ent) for ent in entlist]
                e2t_dict[eid].extend(triple_list)

        for eid in range(self.n_ent):
            e2e_dict[eid].add(eid)
            e2e_dict[eid] = sorted(list(e2e_dict[eid]))
            e2t_dict[eid].append((eid, self.n_rel * 2, eid))
        self.e2e_dict = e2e_dict
        self.e2t_dict = e2t_dict
        print("data preprocess done")

    def data_initialize(self, mode = "e2e"):
        filename = mode
        data_path = os.path.join(self.save_path, filename + '.dat')
        conf_path = os.path.join(self.save_path, filename + '_map.json')
        if not os.path.exists(conf_path):
            e2e_list = []
            e2e_map = {}
            pointer = 0
            if not hasattr(self, filename + "_dict"):
                self.data_preprocess()
            data_dict = getattr(self, filename + "_dict")
            for ent_id, values in data_dict.items():
                e2e_list.extend(values)
                e2e_map[str(ent_id)] = [pointer, len(values)]
                pointer += len(values)
            e2e_list = np.array(e2e_list)
            e2e_map["config"] = [e2e_list.shape, str(e2e_list.dtype)]
            data_mmap = np.memmap(data_path, mode='w+', shape=e2e_list.shape, dtype=e2e_list.dtype)
            data_mmap[:] = e2e_list[:]
            data_mmap.flush()
            save_json(e2e_map, conf_path)
            del data_dict
            delattr(self, filename + "_dict")
        e2e_map = load_json(conf_path)
        data_mmap = np.memmap(data_path, mode='r+', shape=tuple(e2e_map["config"][0]), dtype=e2e_map["config"][1])
        nei_nums = np.array([e2e_map[str(i)][1] for i in range(len(e2e_map) - 1)])
        return data_mmap, e2e_map, nei_nums

    def cache_generate_bak(self, mode):
        nei_data = getattr(self, mode + "_data")
        nei_map = getattr(self, mode + "_map")
        nei_nums = getattr(self, mode + "_nums")
        cache_size = getattr(self, mode + "_cache_size")
        data_item_size = nei_map["config"][0][1] if len(nei_map["config"][0]) > 1 else 1
        item_single_size = np.iinfo(nei_map["config"][1]).bits
        total_data_size = cache_size // (item_single_size * data_item_size)
        sorted_nei_nums = np.argsort(nei_nums * -1)

        current_size, pointer = 0, 0
        data_cache = []
        data_address = np.zeros((len(nei_nums), 2), dtype=np.int32)
        for i in range(len(sorted_nei_nums)):
            nid = sorted_nei_nums[i]
            single_size = nei_nums[nid]
            if current_size + single_size > total_data_size:
                break
            current_size += single_size
            # save this nid
            index, length = nei_map[str(nid)]
            values = nei_data[index:index + length]
            if len(values) != 0:
                data_cache.extend(values.tolist())
                data_address[nid] = [pointer, len(values)]
            pointer += len(values)
        data_cache = np.array(data_cache)
        print(mode+"_cache:", "size:", len(data_cache), "bytes", len(data_cache) * item_single_size * data_item_size, "nodes:", (data_address[:,1] > 0).sum())
        return data_cache, data_address

    def cache_generate(self, sample_big_node=True, ifprint=True):
        e2e_data_item_size = self.e2e_map["config"][0][1] if len(self.e2e_map["config"][0]) > 1 else 1
        e2e_item_single_size = np.iinfo(self.e2e_map["config"][1]).bits
        e2e_total_data_size = self.e2e_cache_size // (e2e_item_single_size * e2e_data_item_size)
        e2t_data_item_size = self.e2t_map["config"][0][1] if len(self.e2t_map["config"][0]) > 1 else 1
        e2t_item_single_size = np.iinfo(self.e2t_map["config"][1]).bits
        e2t_total_data_size = self.e2t_cache_size // (e2t_item_single_size * e2t_data_item_size)
        sorted_nei_nums = np.argsort(self.e2t_nums * -1)

        e2e_nums = copy.deepcopy(self.e2e_nums)
        e2t_nums = copy.deepcopy(self.e2t_nums)

        e2e_current_size, e2e_pointer = 0, 0
        e2t_current_size, e2t_pointer = 0, 0
        e2e_cache, e2t_cache = [], []
        e2e_address = np.zeros((len(self.e2t_nums), 2), dtype=np.int32)
        e2t_address = np.zeros((len(self.e2t_nums), 2), dtype=np.int32)
        for i in range(len(sorted_nei_nums)):
            nid = sorted_nei_nums[i]
            e2t_index, e2t_length = self.e2t_map[str(nid)]
            e2t_values = self.e2t_data[e2t_index:e2t_index + e2t_length]
            if sample_big_node and e2t_length > self.threshold:
                np.random.shuffle(e2t_values)
                rels, relcounts = np.unique(e2t_values[:,1], return_counts=True)
                # print("before", rels, relcounts)
                sorted_ids = np.argsort(relcounts * -1)
                if len(e2t_values) - relcounts[sorted_ids[0]] < self.threshold // 2:
                    rel_mask = e2t_values[:,1]==rels[sorted_ids[0]]
                    sparse_part = e2t_values[~rel_mask]
                    dense_part = e2t_values[rel_mask][:int(self.threshold)-len(sparse_part)]
                    e2t_values = np.concatenate([sparse_part, dense_part], axis=0)
                else:
                    e2t_values = e2t_values[:int(self.threshold)]
                e2e_values = list(set(e2t_values[:,2].tolist()))
            else:
                e2e_index, e2e_length = self.e2e_map[str(nid)]
                e2e_values = self.e2e_data[e2e_index:e2e_index + e2e_length]

            flag1 = e2e_pointer + len(e2e_values) <= e2e_total_data_size
            flag2 = e2t_pointer + len(e2t_values) <= e2t_total_data_size
            if len(e2e_values) != 0 and flag1:
                e2e_cache.extend(e2e_values)
                e2e_address[nid] = [e2e_pointer, len(e2e_values)]
                e2e_pointer += len(e2e_values)
                e2e_nums[nid] = len(e2e_values)
            if len(e2t_values) != 0 and flag2:
                e2t_cache.extend(e2t_values.tolist())
                e2t_address[nid] = [e2t_pointer, len(e2t_values)]
                e2t_pointer += len(e2t_values)
                e2t_nums[nid] = len(e2t_values)
            if not (flag1 or flag2): break
        e2e_cache = np.array(e2e_cache)
        e2t_cache = np.array(e2t_cache)
        self.e2e_cache, self.e2e_address, self.e2e_now_nums = e2e_cache, e2e_address, e2e_nums
        self.e2t_cache, self.e2t_address, self.e2t_now_nums = e2t_cache, e2t_address, e2t_nums

    def subgraph_generate(self, n_layer=2):
        cache_size = self.subG_cache_size
        data_item_size = self.e2t_map["config"][0][1] if len(self.e2t_map["config"][0]) > 1 else 1
        item_single_size = np.iinfo(self.e2t_map["config"][1]).bits
        total_data_size = cache_size // (item_single_size * data_item_size)

        node_mask = (self.e2t_nums > self.threshold)
        node_index = np.arange(len(self.e2e_nums))[node_mask]

        self.subG_cover = np.zeros((len(self.e2e_nums), len(node_index)), dtype=bool)
        self.subG_data = []
        id = 0
        for nid in tqdm(node_index, desc="Subgraph Caching:"):
            covered_edges, covered_nodes = self.load_subgraph([nid], n_layer, remove_big_node=False, with_counts=False)
            self.subG_cover[covered_nodes, id] = True
            self.subG_data.append(covered_edges)
            id += 1
        total_bytes = self.subG_cover.nbytes
        for item in self.subG_data:
            total_bytes += item.nbytes
        print("Amount:", len(self.subG_data), " Bytes:", total_bytes)

    def getAtomNeighbors(self, node_ids):
        node_ids = to_array(node_ids)
        bignode_mask = self.big_node_mask[node_ids]
        if len(node_ids) == 1:
            node_id = node_ids[0]
            address = self.e2e_address[node_id]
            if address[1] > 0 and not (self.readFullData and bignode_mask[0]):
                neis = self.e2e_cache[address[0]:address[0] + address[1]]
            else:
                mapping = self.e2e_map[str(node_id)]
                neis = self.e2e_data[mapping[0]:mapping[0] + mapping[1]]
            return neis
        else:
            node_address = np.array(self.e2e_address)[node_ids]
            node_mask = node_address[:,1] > 0  #先构造mapping_index再统一抽取数据
            cache_index = np.zeros(len(self.e2e_cache), dtype=bool)
            map_index = np.zeros(len(self.e2e_data), dtype=bool)
            for i, node_id in enumerate(node_ids):
                if node_mask[i] and not (self.readFullData and bignode_mask[i]):
                    cache_index[node_address[i][0]:node_address[i][0]+node_address[i][1]] = True
                else:
                    mapping = self.e2e_map[str(node_id)]
                    map_index[mapping[0]:mapping[0]+mapping[1]] = True
            total_neis = self.e2e_cache[cache_index]
            if node_mask.sum() != len(node_ids):
                mapped_neis = self.e2e_data[map_index]
                total_neis = np.concatenate([total_neis, mapped_neis], axis=0)
            return total_neis

    def getAtomTriples(self, node_ids):
        node_ids = to_array(node_ids)
        bignode_mask = self.big_node_mask[node_ids]
        if len(node_ids) == 1 and not (self.readFullData and bignode_mask[0]):
            node_id = node_ids[0]
            address = self.e2t_address[node_id]
            if address[1] > 0:
                return self.e2t_cache[address[0]:address[0] + address[1]]
            else:
                mapping = self.e2t_map[str(node_id)]
                return self.e2t_data[mapping[0]:mapping[0] + mapping[1]]
        else:
            node_address = np.array(self.e2t_address)[node_ids]
            node_mask = node_address[:,1] > 0
            cache_index = np.zeros(len(self.e2t_cache), dtype=bool)
            map_index = np.zeros(len(self.e2t_data), dtype=bool)
            for i, node_id in enumerate(node_ids):
                if node_mask[i] and not (self.readFullData and bignode_mask[i]):
                    cache_index[node_address[i][0]:node_address[i][0]+node_address[i][1]] = True
                else:
                    mapping = self.e2t_map[str(node_id)]
                    map_index[mapping[0]:mapping[0]+mapping[1]] = True
            total_triples = self.e2t_cache[cache_index]
            if node_mask.sum() != len(node_ids):
                mapped_triples = self.e2t_data[map_index]
                total_triples = np.concatenate([total_triples, mapped_triples], axis=0)
            return total_triples

    def getAtomNums(self, node_ids=None, mode="e2e"):
        """
        :param node_ids: required node ids
        :param mode: e2e or e2t
        :param self.readFullData: full_nums or sampled_nums (cache)
        :return: data nums for each node, a list
        """
        data_nums = getattr(self, mode + "_nums") if self.readFullData else getattr(self, mode + "_now_nums")
        return data_nums if node_ids is None else data_nums[node_ids]

    def load_layered_neighborhood(self, batch_nodes, n_layer=3, remove_big_node=True, with_counts=True):
        if with_counts: node_counter = Counter()
        visited_nodes = []
        current_nodes = batch_nodes
        for i in range(n_layer):
            if with_counts: node_counter.update(current_nodes)
            current_nodes = to_array(current_nodes).astype(np.int32)
            nei_list = self.getAtomNeighbors(current_nodes)
            visited_nodes.extend(current_nodes)
            current_nodes = list(set(nei_list) - set(visited_nodes))
        current_nodes.extend(visited_nodes)
        if with_counts:
            node_counter.update(current_nodes)
            node_neis = sorted(node_counter.items(), key=lambda x: x[1], reverse=True)
            return node_neis
        else:
            return current_nodes

    def load_entTriples_withHop(self, batch_nodes, node_hops = None, remove_big_node = True):
        if node_hops is None: node_hops = np.zeros(len(batch_nodes))
        node_triple_list = []
        node_hop_list = []
        for i, node in enumerate(batch_nodes):
            hop_num = node_hops[i]
            triple_parts = self.load_atomTriples(node, remove_big_node)
            hop_parts = [hop_num] * len(triple_parts)
            node_triple_list.extend(triple_parts)
            node_hop_list.extend(hop_parts)
        total_results = np.concatenate([np.array(node_triple_list), np.array(node_hop_list)[:,np.newaxis]], axis=1)
        return total_results

    def load_entTriples(self, batch_nodes, node_hops = None, remove_big_node = True, with_self_edge=True):
        total_results = []
        total_triples = self.load_atomTriples(batch_nodes, remove_big_node, with_self_edge)
        if len(total_triples) > 0:
            total_results = np.concatenate([np.array(total_triples), np.ones([len(total_triples),1])], axis=1)
        return total_results

    def load_atomNeighbors(self, node_ids, remove_big_node = True, with_self_edge = True):
        node_ids = to_array(node_ids).astype(np.int32)
        nei_list = self.getAtomNeighbors(node_ids)
        if with_self_edge:
            nei_list = list(set(nei_list)|set(node_ids))
            nei_list = np.array(nei_list)
        return nei_list

    def load_atomTriples(self, node_ids, remove_big_node = True, with_self_edge = True):
        triples = np.array(self.getAtomTriples(node_ids))
        return triples.tolist()

    def load_neighborhood(self, batch_nodes, n_layer = 3, remove_big_node = True, with_counts = True, verbose=True):
        node_nei_list = []
        if verbose:
            pbar = tqdm(batch_nodes, desc="loadNeis-"+str(n_layer)+":")
        else:
            pbar = batch_nodes
        for node in pbar:
            single_neis = self.load_layered_neighborhood([node], n_layer, remove_big_node, with_counts)
            node_nei_list.append(np.array(single_neis))
        return node_nei_list
    
    def load_structured_neighborhood(self, batch_nodes, n_layer = 3, remove_big_node = True, verbose=True):
        node_nei_list = []
        if verbose:
            pbar = tqdm(batch_nodes, desc="loadNeis-"+str(n_layer)+":")
        else:
            pbar = batch_nodes
        for start_node in pbar:
            visited_nodes = []
            current_nodes = [start_node]
            for i in range(n_layer):
                current_nodes = to_array(current_nodes).astype(np.int32)
                nei_list = self.getAtomNeighbors(current_nodes)
                visited_nodes.extend(current_nodes)
                current_nodes = list(set(nei_list) - set(visited_nodes))
            visited_nodes.extend(current_nodes)
            node_nei_list.append(np.array(visited_nodes))
        return node_nei_list

    def load_subgraph(self, nodes, n_layer=3, remove_big_node=True, with_counts = False):
        current_nodes = nodes if type(nodes) is list else nodes.tolist()
        current_nodes = self.load_layered_neighborhood(current_nodes, n_layer, remove_big_node, with_counts)
        if with_counts:
            current_nodes = np.array(current_nodes).astype(np.int32)
            current_nodes, nei_hops = current_nodes[:,0], current_nodes[:,1]
            node_triples = self.load_entTriples_withHop(current_nodes, nei_hops, remove_big_node)
        else:
            node_triples = self.load_entTriples(current_nodes, remove_big_node)
        node_triples = np.array(node_triples)
        current_nodes = np.array(current_nodes)
        return node_triples, current_nodes