import argparse
from dataset import DataLoader
from models.base import BaseModel
from train import Trainer
from util import *
import torch

torch.set_num_threads(16) # cpu_num

parser = argparse.ArgumentParser(description="Parser for PRINCE")
parser.add_argument('--data_path', type=str, default='data/conceptNet/') #conceptNet  obgl_wikikg2  FB5M  freebase
parser.add_argument('--task_mode', type=str, default='inductive') # allinductive inductive  transductive
parser.add_argument('--model_name', type=str, default='grape') # redgnn, nbfnet, grape
parser.add_argument('--ifcompress', type=bool, default="F")
parser.add_argument('--ifshuffleTrain', type=str, default="F") # total, sub
parser.add_argument('--match_mode', type=int, default=3) # 1,2,3
parser.add_argument('--generate_mode', type=int, default=3)  # 1,2,3
parser.add_argument('--min_capacity', type=float, default=0.9)
parser.add_argument('--system_mode', type=str, default="ours") # basic, ours
parser.add_argument('--num_negative', type=int, default=-1)
parser.add_argument('--max_batch_num', type=int, default=-1)

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--max_epoch', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--n_layer', type=int, default=3) # 2s
parser.add_argument('--superbatch_size', type=int, default=800) # 200
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--sampled_num', type=int, default=1) # 4
parser.add_argument('--slice_size', type=int, default=2048) # 256 2048
# model-related
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--hidden_dim', type=int, default=32) # 64
parser.add_argument('--attn_dim', type=int, default=8)
parser.add_argument('--train_ratio', type=float, default=0.1) # 0.1
parser.add_argument('--remove_one_loop', type=bool, default=False) # True
parser.add_argument('--n_layer2', type=int, default=1)
parser.add_argument('--act', type=str, default='relu')
parser.add_argument('--MESS_FUNC', type=str, default='DistMult') # DistMult RotatE TransE
parser.add_argument('--AGG_FUNC', type=str, default='pna') #sum mean pna
parser.add_argument('--decay_rate', type=float, default=0.8)
parser.add_argument('--lamb', type=float, default=1e-4)
parser.add_argument('--dropout', type=float, default=0.1)
# ginex-related
parser.add_argument('--num_workers', type=int, default=os.cpu_count())
parser.add_argument('--atom_cache_size', type=float, default=200000000)
parser.add_argument('--feature_cache_size', type=float, default=200000000)  # 200000000
parser.add_argument('--big_node_threshold', type=float, default=10000)
parser.add_argument('--trace_load_num_threads', type=int, default=4)
parser.add_argument('--ginex_num_threads', type=int, default=os.cpu_count() * 8)
parser.add_argument('--verbose', dest='verbose', default=False, action='store_true')
parser.add_argument('--save_path', type=str, default='results/')
parser.add_argument('--info', type=str, default='')
args = parser.parse_args()
os.environ['GINEX_NUM_THREADS'] = str(args.ginex_num_threads)

if __name__ == '__main__':
    setup_seed(args.seed)
    args.dataset = args.data_path.split("/")[1]
    kgloader = DataLoader(args.data_path, mode=args.task_mode, feature_cache_size = args.atom_cache_size, big_node_threshold=args.big_node_threshold)
    args.n_rel = kgloader.n_rel
    args.max_batch_num = args.max_batch_num if args.max_batch_num > 0 else None
    args.ifshuffleTrain = True if args.ifshuffleTrain == "T" else False
    args.ifcompress = True if args.ifcompress == "T" else False
    torch.cuda.set_device(args.gpu)
    args.startTimeSpan = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_file_name = "_".join([args.dataset, "M_" + args.model_name]) + "#" + args.startTimeSpan
    checkPath(args.save_path + args.save_file_name)
    logger = init_logger(args)

    model = BaseModel(args)
    trainer = Trainer(args, kgloader, model)
    for epoch_idx in range(args.max_epoch):
        loss = trainer.train_epoch(epoch_idx, ifShuffleTrain=args.ifshuffleTrain, max_batch_num = args.max_batch_num, logger=logger) # 400
        myprint('[Epoch %d] Train Loss:%.4f' % (epoch_idx, loss), logger)
        metrics = trainer.evaluation(epoch_idx, mode="valid", ifpart=True, logger=logger)
        myprint('[Epoch %d] Valid MRR:%.4f H@1:%.4f H@3:%.4f H@10:%.4f' % (epoch_idx, metrics[0], metrics[1], metrics[2], metrics[3]), logger)
        if epoch_idx > 0 and (epoch_idx+1) % 5 == 0:
            metrics = trainer.evaluation(epoch_idx, mode="test", ifpart=False, logger=logger)
            myprint('[Epoch %d] Test MRR:%.4f H@1:%.4f H@3:%.4f H@10:%.4f' % (epoch_idx, metrics[0], metrics[1], metrics[2], metrics[3]), logger)

