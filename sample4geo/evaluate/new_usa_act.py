import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from ..new_trainer import get_query_features, get_similarity, large_test


def CVACT_test(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             cleanup=True):
    query_features, query_labels, grd_gps = get_query_features(config, model, query_dataloader, multi_label=False)
    # torch.save({"feature": query_features, "q_label": query_labels}, "/media/nvidia/E/mkrxp123/Sample4Geo/new_log/query")
    topk_ids, reference_labels = large_test(config, model, reference_dataloader, query_features, query_labels.cpu().numpy())
    # torch.save({"topk_ids": topk_ids, "r_label": reference_labels}, "/media/nvidia/E/mkrxp123/Sample4Geo/new_log/reference")
    
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
    
    for i in tqdm(range(Q)):
        gt_index = ref2index[query_labels_np[i]]
        for ranking, idx in enumerate(topk_ids[i]):
            if idx == gt_index:
                for j, k in enumerate(topk):
                    if ranking < k:
                        results[j] += 1.
                break
            
    results = results / Q * 100.
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
    string.append('Recall@top1%: {:.4f}'.format(results[-1]))
    print(' - '.join(string)) 
    
    if cleanup:
        del query_features, query_labels, grd_gps
        gc.collect()

def calculate_scores(
    similarity      : torch.Tensor, 
    query_labels    : torch.Tensor, 
    reference_labels: torch.Tensor, 
    grd_gps         : torch.Tensor,
    sat_gps         : torch.Tensor,
    ranks           : list[int]=[1,5,10]
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
    
    for i in tqdm(range(Q)):
        # similiarity value of gt reference
        gt_index = ref2index[query_labels_np[i]]
        sim_row = similarity[i].to("cpu")
        gt_sim = sim_row[gt_index]
        
        # number of references with higher similiarity as gt
        higher_sim = sim_row > gt_sim
        
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
                
    results = results / Q * 100.
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
    string.append('Recall@top1%: {:.4f}'.format(results[-1]))
    print(' - '.join(string)) 

    return results[0]


def evaluate(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             cleanup=True):
    
    
    print("\nExtract Features:")
    # reference_features, reference_labels = predict(config, model, reference_dataloader) 
    # query_features, query_labels = predict(config, model, query_dataloader)
    query_features, query_labels, grd_gps = get_query_features(config, model, query_dataloader, multi_label=False)
    similarity, reference_labels, sat_gps = get_similarity(config, model, reference_dataloader, query_features, query_labels.cpu().numpy())
    
    print("Compute Scores:")
    # r1 =  calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
    r1 = calculate_scores(
        # similarity=None,
        similarity=similarity,
        query_labels=query_labels,
        reference_labels=reference_labels,
        grd_gps=grd_gps,
        sat_gps=sat_gps,
        ranks=ranks
    )
    
    # cleanup and free memory on GPU
    if cleanup:
        del reference_labels, query_features, query_labels, similarity
        gc.collect()
        
    return r1


def calculate_nearest(similarity, query_labels, reference_labels, neighbour_range=64):
    # query_labels = query_labels[:,0]
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
    # reference_features, reference_labels = predict(config, model, reference_dataloader) 
    # query_features, query_labels = predict(config, model, query_dataloader)
    query_features, query_labels, grd_gps = get_query_features(config, model, query_dataloader, multi_label=False)
    similarity, reference_labels, sat_gps = get_similarity(config, model, reference_dataloader, query_features)
    
    print("Compute Scores Train:")
    # r1 =  calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
    r1 = calculate_scores(
        similarity=similarity, 
        query_labels=query_labels, 
        reference_labels=reference_labels, 
        grd_gps=grd_gps,
        sat_gps=sat_gps,
        ranks=ranks
    ) 
    
    # near_dict = calculate_nearest(query_features=query_features,
    #                               reference_features=reference_features,
    #                               query_labels=query_labels,
    #                               reference_labels=reference_labels,
    #                               neighbour_range=config.neighbour_range,
    #                               step_size=step_size)
    near_dict = calculate_nearest(
        similarity=similarity,
        query_labels=query_labels,
        reference_labels=reference_labels,
        neighbour_range=config.neighbour_range
    )
            
    # cleanup and free memory on GPU
    if cleanup:
        del reference_labels, query_features, query_labels, similarity
        gc.collect()
        
    return r1, near_dict