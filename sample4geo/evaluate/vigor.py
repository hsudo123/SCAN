import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from ..trainer import predict
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
import h5py
import lmdb
from lz4 import frame

class CachedData(Dataset):
    def __init__(self, config, length):
        super().__init__()
        shape = (length, 1024, config.img_size//32, config.img_size//32)
        self.feature_cache = np.memmap(config.feature_cache, dtype="float32", mode="r+", shape=shape)
        # self.feature_cache = feature_cache
        # self.env = lmdb.open(config.feature_cache, readonly=True, lock=False, readahead=False, meminit=False)
        # self.shape = (1024, config.img_size//32, config.img_size//32)
        
    def __len__(self):
        # with h5py.File(self.feature_cache, "r") as f:
        #     return len(f["feature_maps"])
        return len(self.feature_cache)
    
    def __getitem__(self, index):
        # with h5py.File(self.feature_cache, "r") as f:
        #     return torch.from_numpy(f["feature_maps"][index])
        data = self.feature_cache[index]
        return torch.from_numpy(data)

def reranking(config,
              model,
              query_features,
              reranking_dataloader,
              topk_ids):
    model.eval()
    Q = query_features.size(0)
    reranking_indices = torch.empty(Q, config.topk, dtype=torch.int64)
    
    bar = tqdm(zip(range(Q), reranking_dataloader), total=Q, desc="Reranking")
    for i, sat_map in bar:
        indices, grd_feature = topk_ids[i], query_features[i]
        sat_map = sat_map.to(config.device)
        
        _, sat_reranking = model.attention(grd_feature.unsqueeze(0), sat_map)                           # (topk, dim)
        sim_temp = F.normalize(grd_feature, dim=-1) @ F.normalize(sat_reranking.squeeze(), dim=-1).T    # (topk)
        order = torch.argsort(sim_temp, dim=-1, descending=True, stable=True).cpu()
        reranking_indices[i] = indices[order]
        
    return reranking_indices

def reranking_scores(config, query_labels, reference_labels, reranking_indices, ranks=[1, 5, 10]):
    query_labels = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    results = np.zeros([len(ranks)])
    hit_rate = 0.0
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    for i, indices in enumerate(tqdm(reranking_indices, desc="Calculating_Reranking", leave=False, ncols=140)):
        gt = ref2index[query_labels[i][0]]
        for order, index in enumerate(indices):
            if index == gt:
                for j, rank in enumerate(ranks):   # without top 1%
                    if order < rank:    results[j] += 1
                    
                # hit rate: index ranked before gt must be semi-positive
                hit = True
                semi = [ref2index[label] for label in query_labels[i][1:]]
                for o in range(order-1, -1, -1):
                    if not indices[o] in semi:
                        hit = False
                        break
                if hit: hit_rate += 1.0
                
                break
    
    Q = len(query_labels)     
    results = results / Q * 100.
    hit_rate = hit_rate / Q * 100
    
    string = []
    for i in range(len(ranks)):
        string.append('Recall@{}: {:.4f}'.format(ranks[i], results[i]))
        
    string.append('Hit_Rate: {:.4f}'.format(hit_rate))             
        
    print(' - '.join(string))         
    
    return results[0]

def evaluate(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    model.set_grad_checkpointing(False)
    
    print("\nExtract Features:")
    # Q, R = len(query_dataloader.dataset), len(reference_dataloader.dataset)
    # query_features, query_labels = torch.empty((Q, 1024), dtype=torch.float32, device=config.device), torch.empty((Q, 4), dtype=torch.int64, device=config.device)
    # reference_features, reference_labels = torch.empty((R, 1024), dtype=torch.float32, device=config.device), torch.empty((R,), dtype=torch.int64, device=config.device)
    query_features, query_labels = predict(config, model, query_dataloader, type="grd")
    reference_features, reference_labels = predict(config, model, reference_dataloader, type="sat", memmap=True) 
    
    if config.normalize_features:
        q_features = F.normalize(query_features, dim=-1)
        r_features = F.normalize(reference_features, dim=-1)
    
    print("Compute Scores:")
    r1_retrieval, similarity = calculate_scores(q_features, r_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
    
    topk_scores, topk_ids = torch.topk(similarity, k=config.topk, dim=1)
    del similarity
    reranking_dataset = Subset(CachedData(config, len(reference_dataloader.dataset)), topk_ids.view(-1))
    reranking_dataloader = DataLoader(reranking_dataset,
                                        batch_size=config.topk,
                                        num_workers=config.num_workers,
                                        shuffle=False,
                                        pin_memory=True)
    
    reranking_indices = reranking(config, model, query_features, reranking_dataloader, topk_ids.to("cpu"))
    r1_reranking = reranking_scores(config, query_labels, reference_labels, reranking_indices)
    
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels, reranking_dataset, reranking_dataloader, reranking_indices, topk_scores, topk_ids
        gc.collect()
        
    # return r1_retrieval
    return r1_reranking

def calc_sim(config,
            model,
            reference_dataloader,
            query_dataloader, 
            ranks=[1, 5, 10],
            step_size=1000,
            cleanup=True):
    model.set_grad_checkpointing(False)
    
    print("\nExtract Features:")
    query_features, query_labels = predict(config, model, query_dataloader, type="grd")
    reference_features, reference_labels = predict(config, model, reference_dataloader, type="sat") 
    
    if config.normalize_features:
        query_features = F.normalize(query_features, dim=-1)
        reference_features = F.normalize(reference_features, dim=-1)
    
    print("Compute Scores Train:")
    r1 =  calculate_scores_train(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
    
    near_dict = calculate_nearest(query_features=query_features,
                                  reference_features=reference_features,
                                  query_labels=query_labels,
                                  reference_labels=reference_labels,
                                  neighbour_range=config.neighbour_range,
                                  step_size=step_size)
            
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return r1, near_dict

def calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):
    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    steps = Q // step_size + 1
    
    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    similarity = torch.empty((Q, R), device="cpu", dtype=torch.float32)
    for i in tqdm(range(steps)):
        start = step_size * i
        end = start + step_size
        similarity[start:end] = (query_features[start:end] @ reference_features.T).cpu()

    topk.append(R//100)
    results = np.zeros([len(topk)])
    hit_rate = 0.0
    
    for i in tqdm(range(Q)):
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i][0]]]
        
        # number of references with higher similiarity as gt
        higher_sim = similarity[i,:] > gt_sim
         
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
                        
        # mask for semi pos
        mask = torch.ones(R)
        for near_pos in query_labels_np[i][1:]:
            mask[ref2index[near_pos]] = 0
        
        # calculate hit rate
        hit = (higher_sim * mask).sum()
        if hit < 1:
            hit_rate += 1.0
                
    results = results/ Q * 100.
    hit_rate = hit_rate / Q * 100
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))
    string.append('Hit_Rate: {:.4f}'.format(hit_rate))             
        
    print(' - '.join(string)) 

    return results[0], similarity

def calculate_scores_train(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):

    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    query_labels_np = query_labels[:,0].cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    # similarity = []
    
    # for i in range(steps):
        
    #     start = step_size * i
        
    #     end = start + step_size
          
    #     sim_tmp = query_features[start:end] @ reference_features.T
        
    #     similarity.append(sim_tmp.cpu())
     
    # # matrix Q x R
    # similarity = torch.cat(similarity, dim=0)
    
    similarity = torch.empty((Q, R), device="cpu", dtype=torch.float32)
    for i in tqdm(range(steps)):
        start = step_size * i
        end = start + step_size
        similarity[start:end] = (query_features[start:end] @ reference_features.T).cpu()

    topk.append(R//100)
    
    results = np.zeros([len(topk)])
    
    bar = tqdm(range(Q))
    
    for i in bar:
        
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        
        # number of references with higher similiarity as gt
        higher_sim = similarity[i,:] > gt_sim
         
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
        
    results = results / Q * 100.

    bar.close()
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))           
        
    print(' - '.join(string)) 

    return results[0]
   
def calculate_nearest(query_features, reference_features, query_labels, reference_labels, neighbour_range=64, step_size=1000):

    query_labels = query_labels[:,0]    
    Q = len(query_features)
    R = len(reference_features)
    steps = Q // step_size + 1
    
    # similarity = []
    
    # for i in range(steps):
        
    #     start = step_size * i
        
    #     end = start + step_size
          
    #     sim_tmp = query_features[start:end] @ reference_features.T
        
    #     similarity.append(sim_tmp.cpu())
     
    # # matrix Q x R
    # similarity = torch.cat(similarity, dim=0)
    similarity = torch.empty((Q, R), device="cpu", dtype=torch.float32)
    for i in tqdm(range(steps)):
        start = step_size * i
        end = start + step_size
        similarity[start:end] = (query_features[start:end] @ reference_features.T).cpu()


    # there might be more ground views for same sat view
    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range+2, dim=1)


    topk_references = []
    
    for i in range(len(topk_ids)):
        topk_references.append(reference_labels[topk_ids[i,:]])
    
    topk_references = torch.stack(topk_references, dim=0)

     
    # mask for ids without gt hits
    mask = topk_references != query_labels.unsqueeze(1)
    
    
    topk_references = topk_references.cpu().numpy()
    mask = mask.cpu().numpy()
    

    # dict that only stores ids where similiarity higher than the lowes gt hit score
    nearest_dict = dict()
    
    for i in range(len(topk_references)):
        
        nearest = topk_references[i][mask[i]][:neighbour_range]
    
        nearest_dict[query_labels[i].item()] = list(nearest)
    

    return nearest_dict
