from util import *

class GatherModule(object):
    def __init__(self, args):
        self.args = args

    def initialize(self, mmapped_features, mmapped_filepath, sbatch_featureids, sb_idx):
        effective_sb_size = len(sbatch_featureids)
        self.mmapped_path = mmapped_filepath
        self.num_slices = mmapped_features.shape[0]
        self.num_features = mmapped_features.shape[1]
        self.cache = SliceCache(self.args.feature_cache_size, effective_sb_size, self.num_slices, mmapped_features,
                           self.num_features, sb_idx - 1, self.args.verbose)
        iterptr, iters, initial_cache_indices = self.cache.pass_1_and_2(sbatch_featureids)
        self.changest = self.cache.pass_3(iterptr, iters, initial_cache_indices, sbatch_featureids)
        self.cache.fill_cache(initial_cache_indices, mmapped_features)
        del (initial_cache_indices)

    def gather(self, slice_idx, use_cache=True):
        idx = slice_idx.tolist()
        batch_input = []
        cache_hits = 0
        for i in range(len(idx)):
            if self.cache is not None:
                ifcached_address = self.cache.address_table[idx[i]]
                if use_cache and ifcached_address >= 0:
                    cache_hits += 1
                    batch_input.append(self.cache.cache[ifcached_address])
                else:
                    batch_input.append(self.cache.mmapped_features[idx[i]])
            else:
                batch_input.append(self.mmapped_features[idx[i]])
        batch_input = np.stack(batch_input)
        return batch_input, cache_hits

    def update(self, batch_slice_data, batch_id):
        self.cache.update(batch_slice_data, batch_id)

class SliceCache(object):
    def __init__(self, feature_cache_size, effective_sb_size, num_nodes, mmapped_features,
                 feature_dim, sb, verbose):
        self.size = feature_cache_size
        self.effective_sb_size = effective_sb_size
        if self.effective_sb_size > torch.iinfo(torch.int16).max:
            raise ValueError
        self.num_nodes = num_nodes
        self.mmapped_features = mmapped_features
        self.datatype = mmapped_features.dtype
        self.feature_dim = feature_dim
        self.sb = sb
        self.verbose = verbose

        self.datatype = mmapped_features.dtype
        if self.datatype in [np.uint32, np.int32, np.float32]:
            item_size = 4
        elif self.datatype in [np.uint64, np.int64, np.float64]:
            item_size = 8
        self.num_entries = int(self.size/ item_size / self.feature_dim)
        if self.num_entries > torch.iinfo(torch.int32).max:
            raise ValueError

    def fill_cache(self, indices, mmapped_features):
        self.address_table = torch.full((self.num_nodes,), -1, dtype=torch.int32)
        self.address_table[indices] = torch.arange(indices.numel(), dtype=torch.int32)
        orig_num_threads = torch.get_num_threads()
        torch.set_num_threads(int(os.environ['GINEX_NUM_THREADS']))
        self.cache = mmapped_features[indices.cpu()]
        torch.set_num_threads(orig_num_threads)

    def pass_1_and_2(self, n_id_list):
        frq = torch.zeros(self.num_nodes, dtype=torch.int16)
        filled = False
        count = 0
        initial_cache_indices = torch.empty((0,), dtype=torch.int64, device='cuda')
        for n_id in n_id_list:
            n_id = torch.from_numpy(n_id).long()
            if not filled:
                ta = [frq[n_id] == 0]
                to_cache = n_id[ta].cuda()
                count += to_cache.numel()
                if count >= self.num_entries:
                    to_cache = to_cache[:self.num_entries - (count - to_cache.numel())]
                    initial_cache_indices = torch.cat([initial_cache_indices, to_cache])
                    filled = True
                else:
                    initial_cache_indices = torch.cat([initial_cache_indices, to_cache])
            frq[n_id] += 1

        cumsum = frq.cumsum(dim=0).cuda()
        iterptr = torch.cat([torch.tensor([0, 0], device='cuda'), cumsum[:-1]])
        del (cumsum)
        frq_sum = frq.sum()
        del (frq)

        iters = torch.zeros(frq_sum + 1, dtype=torch.int16, device='cuda')
        iters[-1] = self.effective_sb_size
        msb = (torch.tensor([1], dtype=torch.int16) << 15).cuda()
        # print('msb',msb)

        for i, n_id in enumerate(n_id_list):
            tmp = iterptr[n_id + 1]
            iters[tmp] = i
            del (tmp)
            iterptr[n_id + 1] += 1
            del (n_id)
        iters[iterptr[1:]] |= msb
        iterptr = iterptr[:-1]
        iterptr[0] = 0
        del (n_id_list)

        return iterptr, iters, initial_cache_indices

    def pass_3(self, iterptr, iters, initial_cache_indices, n_id_list):
        effective_sb_size = self.effective_sb_size
        cache_table = torch.zeros(self.num_nodes, dtype=torch.int8, device='cuda')
        cache_table[initial_cache_indices] += 1
        del (initial_cache_indices)

        map_table = torch.full((self.num_nodes,), -1, dtype=torch.int32, device='cuda')
        threshold = 0
        q = []
        msb = (torch.tensor([1], dtype=torch.int16) << 15).cuda()

        for i in range(effective_sb_size):
            n_id = n_id_list[i]
            n_id_cuda = torch.from_numpy(n_id).long().cuda()
            del (n_id)

            map_table[n_id_cuda] = torch.arange(n_id_cuda.numel(), dtype=torch.int32, device='cuda')
            iterptr[n_id_cuda] += 1

            last_access = n_id_cuda[(iters[iterptr[n_id_cuda]] < 0)]
            iterptr[last_access] = iters.numel() - 1;
            del (last_access)

            cache_table[n_id_cuda] += 2
            candidates = (cache_table > 0).nonzero().squeeze()
            del (n_id_cuda)

            next_access_iters = iters[iterptr[candidates]]
            next_access_iters.bitwise_and_(~msb)

            count = (next_access_iters <= threshold).sum()
            prev_status = (count >= self.num_entries)

            if prev_status:
                threshold -= 1
            else:
                threshold += 1
            while (True):
                if threshold > self.effective_sb_size:
                    num_remains = 0
                    break

                count = (next_access_iters <= threshold).sum()
                curr_status = (count >= self.num_entries)
                if (prev_status ^ curr_status):
                    if curr_status:
                        num_remains = self.num_entries - (next_access_iters <= (threshold - 1)).sum()
                        threshold -= 1
                    else:
                        num_remains = self.num_entries - count
                    break
                elif (curr_status):
                    threshold -= 1
                else:
                    threshold += 1

            cache_table[candidates[next_access_iters <= threshold]] |= 4
            cache_table[candidates[next_access_iters == (threshold + 1)][:num_remains]] |= 4
            del (candidates)
            del (next_access_iters)

            in_indices = (cache_table == 2 + 4).nonzero().squeeze()
            in_positions = map_table[in_indices]
            out_indices = ((cache_table == 1) | (cache_table == 3)).nonzero().squeeze()

            cache_table >>= 2
            map_table[:] = -1
            q.append([in_indices, out_indices, in_positions])

            del (in_indices)
            del (out_indices)
            del (in_positions)

        del (cache_table)
        del (iterptr)
        del (iters)
        del (map_table)

        self.changeset = q
        return q

    def update(self, batch_inputs, batch_id):
        [in_indices, out_indices, in_positions] = self.changeset[batch_id]
        if (in_indices.shape == torch.Size([])):
            # print("fix 1")
            in_indices = torch.tensor([in_indices])
            in_positions = torch.tensor([in_positions])
        if (out_indices.shape == torch.Size([])):
            # print("fix 2")
            out_indices = torch.tensor([out_indices])
        if (len(in_indices) > len(out_indices)):
            # print("fix 3")
            in_indices = in_indices[:len(out_indices)]
            in_positions = in_positions[:len(out_indices)]
        elif (len(in_indices) < len(out_indices)):
            # print("fix 4")
            out_indices = out_indices[:len(in_indices)]
        in_indices = in_indices.cpu()
        in_positions = (in_positions.cpu()).type(torch.long)
        out_indices = out_indices.cpu()
        cache_out_idx = (self.address_table[out_indices]).type(torch.long)
        self.cache[cache_out_idx] = batch_inputs[in_positions]
        self.address_table[in_indices] = cache_out_idx.type(torch.int32)
        self.address_table[out_indices] = -1
