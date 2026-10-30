import cv2
import time
import torch
import numpy as np
from tqdm import tqdm
from .utils import AverageMeter
from torch.amp import autocast
# from torch.cuda.amp import autocast
import torch.nn.functional as F
from .mercator import get_latlon_tensor
import copy

def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None, writer=None, curr_step=0, fixed=False):
    # set model train mode
    model.train()
    use_bev = train_config.use_bev
    if train_config.grad_checkpointing: model.set_grad_checkpointing(True)
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for i, (query, query_bev, reference, gt_index) in enumerate(bar):
    # for i, (query, reference, gt_index) in enumerate(bar):
        if scaler:
            with autocast(device_type="cuda"):
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
            
                # Forward pass
                if use_bev:
                    grd_pano_feature, grd_bev_feature, sat_pano_feature, sat_bev_feature, attention_weight = \
                        model.train_forward(query, query_bev.to(train_config.device), reference)
                else:
                    grd_pano_feature, sat_pano_feature, attention_weight = model(query, reference)
                    grd_bev_feature, sat_bev_feature = None, None
                    
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                    temperature = model.module.temperature.exp()
                else:
                    temperature = model.temperature.exp()
                
                retreival_loss, pano_bev_loss, bev_scan_loss = loss_function(
                    grd_pano_feature, 
                    sat_pano_feature,
                    grd_bev_feature,
                    sat_bev_feature,
                    attention_weight,
                    gt_index.to(train_config.device),
                    temperature
                )
                
                loss = retreival_loss 
                if use_bev: 
                    loss += train_config.bev_coeff * (pano_bev_loss + bev_scan_loss)
                    writer.add_scalar("Train/pano_bev_loss", pano_bev_loss.item(), curr_step+i)
                    writer.add_scalar("Train/bev_scan_loss", bev_scan_loss.item(), curr_step+i)
                
                loss_val = loss.item()
                losses.update(loss_val)
                # writer.add_scalar("Train/loss", loss_val, curr_step+i)
                writer.add_scalar("Train/lr", scheduler.get_last_lr()[0], curr_step+i)
                writer.add_scalar("Train/retreival_loss", retreival_loss.item(), curr_step+i)
                writer.add_scalar("Train/retreival_temp", temperature[0].item(), curr_step+i)
                # writer.add_scalar("Train/reranking_loss", reranking_loss.item(), curr_step+i)
                # writer.add_scalar("Train/reranking_temp", temperature[1].item(), curr_step+i)
                # writer.add_scalar("Train/position_loss", position_loss.item(), curr_step+i)

            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if scheduler is not None:
                scheduler.step()
            # if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
            #     scheduler.step()
   
        else:
        
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            grd_retrieval, sat_retrieval = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_function(grd_retrieval, sat_retrieval, model.module.logit_scale.exp())
            else:
                loss = loss_function(grd_retrieval, sat_retrieval, model.logit_scale.exp()) 
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
              
        if train_config.verbose:
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg

@torch.no_grad()
def get_query_features(config, model, dataloader, multi_label: bool=True):
    model.eval()
    
    curr, all = 0, len(dataloader.dataset)
    # extractor = model.grd_extractor
    img_features = torch.empty((all, config.latent_dim), device=config.device, dtype=torch.float32)
    grd_gps = torch.empty((all, 2), device="cpu", dtype=torch.float32)
    ids = torch.empty((all, 4), device="cpu", dtype=torch.int64) if multi_label else torch.empty((all,), device="cpu", dtype=torch.int64)
    
    for img, id, gps in tqdm(dataloader, total=len(dataloader)):
        # ids_list.append(id)
        length = len(id)
        ids[curr:curr+length] = id
        grd_gps[curr:curr+length] = gps
        
        with autocast(device_type="cuda"):
            img = img.to(config.device)
            # img_feature = extractor(img)
            img_feature = model.pano_aggregator(model.grd_extractor(img))
        
        # save features in fp32 for sim calculation
        # img_features_list.append(img_feature.to(torch.float32))
        img_features[curr:curr+length] = img_feature.to(torch.float32)
        curr += length
        
    ids = ids.to(config.device)
        
    return img_features, ids, grd_gps

@torch.no_grad()
def get_similarity(config, model, dataloader, query_features, query_labels_np: np.ndarray=None):
    model.eval()

    curr = 0
    Q, _ = query_features.shape
    R = len(dataloader.dataset)
    step_size = config.batch_size_eval
    steps = Q // step_size + 1
    # sat_size = config.sat_size
    # map_size = config.img_size // 32
    
    # extractor = model.sat_extractor
    # sat_gps = torch.empty((R, 2), device="cpu", dtype=torch.float32)
    ids = torch.empty((R,), device="cpu", dtype=torch.int64)
    sat_gps = torch.empty((Q, 2), device="cpu", dtype=torch.float32)
    
    similarity = torch.empty((Q,R), device=config.buffer_device, dtype=torch.float32)
    for i, (img, id, gps) in enumerate(tqdm(dataloader, total=len(dataloader))):
        length = len(id)
        ids[curr:curr+length] = id
        # sat_gps[curr:curr+length] = gps
        # temp_coords = torch.empty((Q, length, 2), device=config.buffer_device, dtype=torch.float32)
        # temp_maps = torch.empty((Q, length, map_size, map_size), device=config.buffer_device, dtype=torch.float32)
        
        with autocast(device_type="cuda"):
            img = img.to(config.device)
            # reference_map = extractor(img)
            reference_map = model.sat_extractor(img)
            
            for j in range(steps):
                start = step_size * j
                end = start + step_size
                attention_map, reference_feature = model.attention(query_features[start:end], reference_map)
                
                # temp_maps[start:end] = attention_map.to(config.buffer_device)
                # q, r, h, w = attention_map.shape
                # position = attention_map.view(q, r, -1).argmax(dim=-1)
                # row, col = position // w, position % w  # (q, r)
                # temp_coords[start:end] = torch.stack([sat_size * (row + 0.5) / h, sat_size * (col + 0.5) / w], dim=-1).to(config.buffer_device)
                # similarity[start:end] = (query_features[start:end] @ reference_features.T).cpu().numpy()
                sim_temp = torch.einsum(
                    "GD,GSD->GS",
                    F.normalize(query_features[start:end], dim=-1),
                    F.normalize(reference_feature.to(torch.float32), dim=-1)
                )
                similarity[start:end, curr:curr+length] = sim_temp.to(config.buffer_device)
                
        curr += length
        
        # if query_labels_np is None:
        #     continue
        # for j, sat_label_tensor in enumerate(id):
        #     sat_label = sat_label_tensor.item()
        #     for k, grd_label in enumerate(query_labels_np):
        #         if grd_label == sat_label:
        #             # distribution = cv2.resize(temp_maps[k, j].cpu().numpy(), dsize=(sat_size, sat_size), interpolation=cv2.INTER_CUBIC).flatten()
        #             # argmax = torch.tensor(np.argmax(distribution))
        #             # u, v = argmax // sat_size, argmax % sat_size
        #             u, v = temp_coords[k, j].to("cpu")
        #             sat_gps[k, 0], sat_gps[k, 1] = get_latlon_tensor(gps[j, 0], gps[j, 1], u, v, 20, map_size)
        #             # sat_gps[k, 0], sat_gps[k, 1] = get_latlon_tensor(gps[j, 0], gps[j, 1], 320, 320, 20, sat_size)
                
        # for j, idx_tensor in enumerate(id):
        #     # print(idx)
        #     idx = idx_tensor.item()
        #     if idx in qry_keys:
        #         gt_index = qry2index[idx]
        #         u, v = temp_coords[gt_index, j].to("cpu")
        #         # print(gps[j, 0], gps[j, 1], u, v)
        #         sat_gps[gt_index, 0], sat_gps[gt_index, 1] = get_latlon_tensor(gps[j, 0], gps[j, 1], u, v, 20, sat_size)

    ids = ids.to(config.device)
        
    return similarity, ids, sat_gps


@torch.no_grad()
def large_test(config, model, dataloader, query_features, query_labels_np: np.ndarray=None, ranks: list[int]=[1,5,10]):
    model.eval()

    curr = 0
    Q, _ = query_features.shape
    R = len(dataloader.dataset)
    step_size = config.batch_size_eval
    steps = Q // step_size + 1
    ids = torch.empty((R,), device="cpu", dtype=torch.int64)
    query_features_normalized = F.normalize(query_features, dim=-1)
    
    # ref2index = {label: idx for idx, label in dataloader.dataset.idx2num_idx.items()}
    # topk = copy.deepcopy(ranks)
    # topk.append(R//100)
    # results = np.zeros([len(topk)])
    
    topk_ids, topk_sims = torch.empty((Q, 0), dtype=torch.long, device="cpu"), torch.empty((Q, 0), dtype=torch.float32, device=config.buffer_device)
    
    for i, (img, id, gps) in enumerate(tqdm(dataloader, total=len(dataloader))):
        length = len(id)
        ids[curr:curr+length] = id
        
        sim_part = torch.empty((Q, length), dtype=torch.float32, device=config.buffer_device)
        
        with autocast(device_type="cuda"):
            img = img.to(config.device)
            reference_map = model.sat_extractor(img)
            
            for j in range(steps):
                start = step_size * j
                end = start + step_size
                attention_map, reference_feature = model.attention(query_features[start:end], reference_map)
                
                sim_temp = torch.einsum(
                    "GD,GSD->GS",
                    query_features_normalized[start:end],
                    F.normalize(reference_feature.to(torch.float32), dim=-1)
                )
                sim_part[start:end] = sim_temp.to(config.buffer_device)
        
        new_sims = torch.concat([topk_sims, sim_part], dim=1)
        topk_sims, topk_indices = torch.topk(
            new_sims, 
            min(R//100, new_sims.shape[1]), 
            dim=1, largest=True, sorted=True
        )
        
        topk_ids = torch.gather(
            torch.concat([topk_ids, torch.arange(curr, curr+length).unsqueeze(0).expand(Q, -1)], dim=1), 
            dim=1, index=topk_indices.to("cpu")
        )
        curr += length
        
    return topk_ids, ids
        
    # for i, q_feature_part in enumerate(tqdm(query_features.chunk(config.batch_size_eval))):
    #     length = len(q_feature_part)
    #     sim_rows = torch.empty((length, R), dtype=torch.float32, device=config.buffer_device)
    #     q_start = step_size * i
    #     q_end = q_start + step_size
    #     # curr = 0
    #     for j, (img, id, _) in enumerate(dataloader):
    #         r_start = step_size * j
    #         r_end = r_start + step_size
    #         # ids[start:end] = id
            
    #         with autocast(device_type="cuda"):
    #             img = img.to(config.device)
    #             reference_map = model.sat_extractor(img)
    #             attention_map, reference_feature = model.attention(q_feature_part, reference_map)

    #             sim_temp = torch.einsum(
    #                 "GD,GSD->GS",
    #                 F.normalize(q_feature_part, dim=-1),
    #                 F.normalize(reference_feature.to(torch.float32), dim=-1)
    #             )
    #             sim_rows[:, r_start:r_end] = sim_temp
    #         # curr += length
                
    #     for j, sim_row in enumerate(sim_rows):
    #         sim_row = sim_row.to("cpu")
    #         gt_index = ref2index[query_labels_np[q_start+j]]
    #         print(gt_index)
    #         gt_sim = sim_row[gt_index]
            
    #         higher_sim = sim_row > gt_sim

    #         ranking = higher_sim.sum()
    #         for j, k in enumerate(topk):
    #             if ranking < k:
    #                 results[j] += 1.
    
    # string = []
    # for j in range(len(topk)-1):
    #     string.append('Recall@{}: {:.4f}'.format(topk[j], results[j]))
    # string.append('Recall@top1%: {:.4f}'.format(results[-1]))
    # print(' - '.join(string)) 