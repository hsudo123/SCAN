import gc
import copy
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from ..new_trainer import get_query_features, get_similarity
from ..mercator import get_latlon_tensor, gps2distance
import matplotlib.pyplot as plt

def calculate_scores(
    similarity      : torch.Tensor, 
    query_labels    : torch.Tensor, 
    reference_labels: torch.Tensor, 
    grd_gps         : torch.Tensor,
    sat_gps         : torch.Tensor,
    ranks           : list[int]=[1,5,10], 
    full_eval       : bool=True
):
    topk = copy.deepcopy(ranks)
    # Q, R = similarity.shape
    Q, R = len(query_labels), len(reference_labels)
    
    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i

    topk.append(R//100)
    results = np.zeros([len(topk)])
    hit_rate = 0.0
    # distance = torch.empty(Q, device="cpu", dtype=torch.float32)
    # distance = []
    
    for i in tqdm(range(Q)):
        # similiarity value of gt reference
        # gt_sim = similarity[i, ref2index[query_labels_np[i][0]]]
        gt_index = ref2index[query_labels_np[i][0]]
        sim_row = similarity[i].to("cpu")
        gt_sim = sim_row[gt_index]
        
        # number of references with higher similiarity as gt
        higher_sim = sim_row > gt_sim
        
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
        
        if full_eval:               
            # mask for semi pos
            mask = np.ones(R)
            for near_pos in query_labels_np[i][1:]:
                mask[ref2index[near_pos]] = 0
            
            # calculate hit rate
            hit = (higher_sim * mask).sum()
            if hit < 1:
                hit_rate += 1.0
            
                # est_lat, est_lon = sat_gps[i].to("cpu")
                # # distance[i] = gps2distance(grd_gps[i, 0], grd_gps[i, 1], est_lat, est_lon)
                # distance.append(gps2distance(grd_gps[i, 0], grd_gps[i, 1], est_lat, est_lon))
            
            # est_lat, est_lon = get_latlon_tensor(sat_gps[gt_index, 0], sat_gps[gt_index, 1], x[:, 0], x[:, 1], 20)
            # epe = gps2distance(grd_gps[i, 0], grd_gps[i, 1], est_lat, est_lon)
            # distance = { # for people
            #     'epe': epe.nanmean().item(), 
            #     '1px': (epe < 1).float().mean().item(), # R@1
            #     '3px': (epe < 3).float().mean().item(),
            #     '5px': (epe < 5).float().mean().item(),
            # }
                
    results = results / Q * 100.
    hit_rate = hit_rate / Q * 100
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
    string.append('Recall@top1%: {:.4f}'.format(results[-1]))
    string.append('Hit_Rate: {:.4f}'.format(hit_rate))  
    print(' - '.join(string)) 
    # if full_eval:
    #     distance = torch.Tensor(distance)
    #     print({ # for people
    #         'epe': distance.nanmean().item(), 
    #         '1px': (distance < 1).float().mean().item(), # R@1
    #         '3px': (distance < 3).float().mean().item(),
    #         '5px': (distance < 5).float().mean().item(),
    #     })

    return results[0]

def evaluate(
    config,
    model,
    reference_dataloader,
    query_dataloader, 
    ranks=[1, 5, 10],
    cleanup=True
):
    model.set_grad_checkpointing(False)
    
    print("\nExtract Features:")
    # Q, R = len(query_dataloader.dataset), len(reference_dataloader.dataset)
    # query_features, query_labels = torch.empty((Q, config.hidden_dim), dtype=torch.float32, device=config.device), torch.empty((Q, 4), dtype=torch.int64, device=config.device)
    query_features, query_labels, grd_gps = get_query_features(config, model, query_dataloader)
    # torch.save({
    #     "query_features": query_features,
    #     "query_labels": query_labels,
    #     "grd_gps": grd_gps
    # }, "query")
    # qry = torch.load("query", map_location="cpu")
    # grd_gps = qry["grd_gps"]
    # query_labels = qry["query_labels"]
    # query_features = qry["query_features"].to(config.device)
    similarity, reference_labels, sat_gps = get_similarity(config, model, reference_dataloader, query_features, query_labels[:, 0].cpu().numpy())
    # torch.save({
    #     # "query_features": query_features,
    #     "reference_labels": reference_labels,
    #     "sat_gps": sat_gps
    # }, "reference")
    # ref = torch.load("reference", map_location="cpu")
    # sat_gps = ref["sat_gps"]
    # reference_labels = ref["reference_labels"]
    # print(grd_gps)
    # print(sat_gps)
    
    r1 = calculate_scores(
        # similarity=None,
        similarity=similarity,
        query_labels=query_labels,
        reference_labels=reference_labels,
        grd_gps=grd_gps,
        sat_gps=sat_gps,
        ranks=ranks,
        full_eval=True
    )
    
    if cleanup:
        del reference_labels, query_labels, query_features, similarity
        gc.collect()
        
    return r1

def calculate_scores_train(similarity, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):

    topk = copy.deepcopy(ranks)
    Q, R = similarity.shape
    
    steps = Q // step_size + 1
    
    query_labels_np = query_labels[:,0].cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i

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

def calculate_nearest(similarity, query_labels, reference_labels, neighbour_range=64):
    query_labels = query_labels[:,0]
    Q, R = similarity.shape

    # there might be more ground views for same sat view
    topk_scores, topk_ids = torch.topk(similarity.to("cpu"), k=neighbour_range+2, dim=1)

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
    
def calc_sim(
    config,
    model,
    reference_dataloader,
    query_dataloader, 
    ranks=[1, 5, 10],
    cleanup=True
):
    model.set_grad_checkpointing(False)
    
    print("\nExtract Features:")
    # Q, R = len(query_dataloader.dataset), len(reference_dataloader.dataset)
    # query_features, query_labels = torch.empty((Q, config.hidden_dim), dtype=torch.float32, device=config.device), torch.empty((Q, 4), dtype=torch.int64, device=config.device)
    query_features, query_labels, grd_gps = get_query_features(config, model, query_dataloader)
    similarity, reference_labels, sat_gps = get_similarity(config, model, reference_dataloader, query_features)
    
    print("Compute Scores Train:")
    r1 = calculate_scores(
        similarity=similarity, 
        query_labels=query_labels, 
        reference_labels=reference_labels, 
        grd_gps=grd_gps,
        sat_gps=sat_gps,
        ranks=ranks,
        full_eval=False
    ) 
    
    near_dict = calculate_nearest(
        similarity=similarity,
        query_labels=query_labels,
        reference_labels=reference_labels,
        neighbour_range=config.neighbour_range
    )
            
    # cleanup and free memory on GPU
    if cleanup:
        del reference_labels, query_labels, query_features, similarity
        gc.collect()
        
    return r1, near_dict