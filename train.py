import gc
from tqdm import tqdm
from util import *
from modules.sampler import SamplerModule
from modules.slicer import SlicerModule
from modules.gather import GatherModule
from modules.compress import Compresser
from models.base import SubGraph

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

class Trainer(object):
    def __init__(self, args, loader, model=None):
        self.args = args
        self.loader = loader
        self.model = model
        self.batch_size = args.batch_size
        self.superbatch_size = args.superbatch_size
        self.slice_size = args.slice_size
        self.n_layer = args.n_layer
        self.sampler = SamplerModule(args)
        self.slicer = SlicerModule(args)
        self.gather = GatherModule(args)
        self.loader.load_before_train(showTime=True)
        self.loader.load_graph_process(showTime=True)
        self.compressor = Compresser(self.slice_size, self.loader.n_ent, self.loader.n_rel)
        self.slicer.initialize(loader, self.compressor)

    def train_epoch(self, epoch_id, showTime=True, ifShuffleTrain=True, max_batch_num = None, logger=None):
        if ifShuffleTrain:
            self.loader.shuffle_train(ratio=self.args.train_ratio, remove_one_loop=self.args.remove_one_loop)  # 0.1
        else:
            np.random.shuffle(self.loader.train_data)
        self.model.model.train()
        n_epoch_train = self.loader.n_train
        n_batch = get_batch_num(n_epoch_train, self.batch_size)
        n_sbatch = get_batch_num(n_batch, self.superbatch_size)
        myprint('[Train Epoch %d] train_triples: %d total_batchs: %d total_superbatchs: %d' % (epoch_id, n_epoch_train, n_batch, n_sbatch), logger)
        print("#"*40)
        total_loss = 0
        for sb_idx in range(n_sbatch):
            print("start new superbatch", "#"*20)
            sb_loss = self.superbatch_excute(epoch_id, sb_idx, n_epoch_train,
                                   mode="train", showTime=showTime, logger=logger)
            total_loss += sb_loss
        return total_loss

    def evaluation(self, epoch_id, showTime=True, mode="test", ifpart=False, logger=None):
        self.loader.load_before_evaluate(showTime)
        self.loader.load_graph_process(showTime)
        self.model.model.eval()
        total_ranking, total_relation = [], []
        n_evaluate_data = self.loader.n_valid if mode == "valid" else self.loader.n_test
        if ifpart: n_evaluate_data = min(5000, n_evaluate_data)
        n_batch = get_batch_num(n_evaluate_data, self.batch_size)
        n_sbatch = get_batch_num(n_batch, self.superbatch_size)
        myprint('[Evaluate %s] eval_triples: %d total_batchs: %d total_superbatchs: %d' % (mode, n_evaluate_data, n_batch, n_sbatch), logger)
        #print("Evaluate", mode, ": eval_triples", n_evaluate_data, "total_batchs", n_batch, "total_superbatchs", n_sbatch)
        print("#" * 40)
        for sb_idx in range(n_sbatch):
            print("start new superbatch", "#" * 20)
            ranking, relaing = self.superbatch_excute(epoch_id, sb_idx, n_evaluate_data,
                                             mode=mode, showTime=showTime, logger=logger)
            total_ranking.extend(ranking)
            total_relation.extend(relaing)
        total_ranking, total_relation = np.array(total_ranking), np.array(total_relation)
        v_mrr, v_h1, v_h3, v_h10 = cal_performance(total_ranking)
        myprint('[Total %s] MRR:%.4f H@1:%.4f H@3:%.4f H@10:%.4f'%(mode, v_mrr, v_h1, v_h3, v_h10), logger)
        return v_mrr, v_h1, v_h3, v_h10

    def superbatch_excute(self, epoch_id, sb_idx, n_data_num, mode="train", showTime=True, logger=None):
        #################################################################
        # 1. sample triple data for multiple batchs
        if mode == "train":
            sbatch_input_triples, start_nodes = \
                self.sampler.basic_sampling(self.loader, self.batch_size,
                                            self.superbatch_size, n_data_num, sb_idx, mode)
        elif mode in ["valid", "test"]:
            sbatch_input_triples, sbatch_label_csrmat, start_nodes = \
                self.sampler.eval_unsampling(self.loader, self.batch_size,
                                             self.superbatch_size, n_data_num, sb_idx, mode="test")
        n_start_nodes = len(start_nodes)
        #################################################################
        # 2. generate node subgraph and slices
        previous_slice_num = self.slicer.total_slice_num
        redundancy_rate, reuse_rate = self.slicer.slice_nodes(start_nodes, showTime=True) #*******
        self.mmapped_slices = self.slicer.get_slicer_data()

        sbatch_train_slices, sbatch_train_n2smat = [], []
        for b_idx, batch_triples in enumerate(sbatch_input_triples):
            batch_slices, batch_n2smat = self.slicer.get_slice_list(batch_triples[:, 0])
            sbatch_train_slices.append(batch_slices)
            sbatch_train_n2smat.append(batch_n2smat)
        #################################################################
        # 3. running each batch
        self.gather.initialize(self.mmapped_slices, self.slicer.slice_file, sbatch_train_slices, sb_idx)
        num_iter = len(sbatch_train_slices)
        total_cache_hits, total_slice_num = 0, 0
        if mode == "train":
            total_batch_loss = 0
        else:
            ranking, relaing = [], []
        pbar = tqdm(range(num_iter), desc="BatchRun:")
        for b_idx in pbar:
            batch_triples = sbatch_input_triples[b_idx]

            batch_slices = sbatch_train_slices[b_idx]
            batch_n2smat = sbatch_train_n2smat[b_idx]
            if mode != "train":
                batch_labcsr = sbatch_label_csrmat[b_idx]

            batch_slice_data, batch_cache_hits = self.gather.gather(batch_slices, use_cache=True)
            total_slice_num += len(batch_slices)
            total_cache_hits += batch_cache_hits
            #################################################################
            if self.slicer.ifcompress:
                total_triples = self.compressor.multi_decode(batch_slice_data)
                torch_triples = torch.from_numpy(total_triples).long().cuda()
                torch_triples = torch.unique(torch_triples, dim=0)
            else:
                batch_slice_triples = batch_slice_data.reshape(len(batch_slices), self.args.slice_size, -1)
                torch_triples = torch.from_numpy(batch_slice_triples).reshape(len(batch_slices) * self.args.slice_size, -1)
                torch_triples = torch_triples[torch_triples[:, -1] != 0].cuda()
                torch_triples = torch.unique(torch_triples[:,:3], dim=0)

            error_slice_list = []
            for start_node in batch_triples[:,0]:
                if (torch_triples[:,0] == start_node).sum() == 0:
                    error_slices, _ = self.slicer.get_slice_list([int(start_node)])
                    error_slice_list.extend(error_slices)
            error_slices = np.array(list(set(error_slice_list)))
            if len(error_slice_list) > 0:
                error_slice_data, error_cache_hits = self.gather.gather(error_slices, use_cache=False)
                if self.slicer.ifcompress:
                    error_triples = self.compressor.multi_decode(error_slice_data)
                    error_triples = torch.from_numpy(error_triples).long().cuda()
                else:
                    error_slice_triples = error_slice_data.reshape(len(error_slices), self.args.slice_size, -1)
                    error_triples = torch.from_numpy(error_slice_triples).reshape(len(error_slices) * self.args.slice_size,-1)
                    error_triples = error_triples[error_triples[:, -1] != 0][:, :3].cuda()
                torch_triples = torch.concat([torch_triples, error_triples], dim=0)
                torch_triples = torch.unique(torch_triples, dim=0)

            unique_ents = torch.unique(torch_triples[:, 2], dim=0)  # 2
            self_triples = torch.stack(
                [unique_ents, torch.ones(len(unique_ents)).cuda() * (self.loader.n_rel * 2), unique_ents,
                 torch.ones(len(unique_ents)).cuda() * self.args.n_layer], dim=1)
            torch_filtered_triples = torch.concat([torch_triples[:, :3], self_triples[:, :3].long()], dim=0)
            torch_filtered_triples = torch.unique(torch_filtered_triples, dim=0)

            sampled_num = self.args.sampled_num
            if mode == "train" and sampled_num > 1:
                extend_batch_triples = []
                for nid in batch_triples[:,0]:
                    triples = self.loader.load_atomTriples(nid, with_self_edge=False)
                    triples = np.array(triples)
                    sampled_ids = np.random.randint(0,len(triples),size=[sampled_num])
                    extend_batch_triples.append(triples[sampled_ids])
                    del triples
                extend_batch_triples = np.concatenate(extend_batch_triples, axis=0)
                batch_triples = extend_batch_triples

            subgraph = SubGraph(self.args, torch_filtered_triples)
            batch_retriples = subgraph.encode(batch_triples)
            pbar.set_postfix(subE = len(subgraph.Ent), subT = len(subgraph.KG))

            if mode == "train":
                batch_loss = self.model.train_batch(batch_retriples, subgraph, epoch_id)
                total_batch_loss += batch_loss
            else:
                batch_retriples[:,2] = batch_triples[:,0]
                batch_ranks, batch_rels = self.model.eval_batch(batch_retriples, subgraph, batch_labcsr,
                                                    self.loader.filters, self.loader.n_ent,
                                                    self.args.num_negative, epoch=epoch_id)
                ranking.extend(batch_ranks)
                relaing.extend(batch_rels)
            #################################################################
            if (b_idx != num_iter - 1):
                self.gather.update(batch_slice_data, b_idx)

        if hasattr(self.slicer, "directNei_dict"): del self.slicer.directNei_dict
        self.slicer.directNei_dict = {}
        torch.cuda.empty_cache()
        gc.collect()
        if mode == "train":
            myprint('[%d] Superbatch loss:%.4f total_slices:%d' % (sb_idx, total_batch_loss, self.slicer.total_slice_num), logger)
            return total_batch_loss
        else:
            return ranking, relaing
