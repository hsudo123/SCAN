import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.amp import autocast
# from torch.cuda.amp import autocast
import torch.nn.functional as F
import numpy as np
import h5py
import lmdb
from lz4 import frame

def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None, writer=None, curr_step=0, fixed=False):

    # set model train mode
    model.train()
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
    for i, (query, reference, gt_index) in enumerate(bar):
        
        if scaler:
            with autocast(device_type="cuda"):
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
            
                # Forward pass
                grd_retrieval, sat_retrieval, sat_reranking, attention_weight = model(query, reference)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    # loss = loss_function(features1, features2, model.module.logit_scale.exp())
                    temperature = model.module.temperature.exp()
                else:
                    # loss = loss_function(features1, features2, model.logit_scale.exp()) 
                    temperature = model.temperature.exp()
                
                retreival_loss, reranking_loss, position_loss = loss_function(
                    grd_retrieval, 
                    sat_retrieval,
                    sat_reranking,
                    attention_weight,
                    gt_index.to(train_config.device),
                    # [1 / 0.07, 1 / 0.07] if fixed else temperature
                    temperature
                ) 
                loss = retreival_loss + \
                       train_config.reranking_coeff * reranking_loss + \
                       train_config.position_coeff * position_loss
                
                if not fixed:   
                    loss += train_config.temperature_coeff * (temperature[0] + temperature[1])
                    
                loss_val = loss.item()
                losses.update(loss_val)
                writer.add_scalar("Train/loss", loss_val, curr_step+i)
                writer.add_scalar("Train/retreival_loss", retreival_loss.item(), curr_step+i)
                writer.add_scalar("Train/retreival_temp", temperature[0].item(), curr_step+i)
                writer.add_scalar("Train/reranking_loss", reranking_loss.item(), curr_step+i)
                writer.add_scalar("Train/reranking_temp", temperature[1].item(), curr_step+i)
                writer.add_scalar("Train/position_loss", position_loss.item(), curr_step+i)
                
                  
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
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
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
def predict(train_config, model, dataloader, type="grd", memmap=False):
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    # img_features_list = []
    # ids_list = []
    
    all = len(dataloader.dataset)
    shape = (all, 4) if type == "grd" else (all,)
    extractor = model.grd_extractor if type == "grd" else model.sat_extractor
    
    img_features = torch.empty((all, 1024), device=train_config.device, dtype=torch.float32)
    ids = torch.empty(shape, device="cpu", dtype=torch.int64)
    curr = 0
    
    # if memmap:
    #     max_size = 100 * 1024 * 1024 * 1024
    #     env = lmdb.open(train_config.feature_cache, map_size=max_size)
        
    #     txn = env.begin(write=True)
    #     txn.put(b"length", str(all).encode())
        
    #     for idx, (img, id) in enumerate(bar):
    #         # ids_list.append(id)
    #         length = len(id)
    #         ids[curr:curr+length] = id
            
    #         with autocast(device_type="cuda"):
    #             img = img.to(train_config.device)
    #             img_feature, img_map = extractor.extract(img)
                
    #             for j in range(len(img_map)):
    #                 raw_bytes = img_map[j].cpu().numpy().tobytes()
    #                 compressed = frame.compress(raw_bytes)
    #                 txn.put(f"{curr+j}".encode(), compressed)
                
    #         if (idx + 1) % 25 == 0:
    #             txn.commit()
    #             txn = env.begin(write=True)
                
    #         # save features in fp32 for sim calculation
    #         # img_features_list.append(img_feature.to(torch.float32))
    #         img_features[curr:curr+length] = img_feature.to(torch.float32)
    #         curr += length
        
    #     txn.commit()  # 最後剩下沒滿1000的，也要commit一下
    #     env.close()
    # else :
    #     for img, id in bar:
    #         # ids_list.append(id)
    #         length = len(id)
    #         ids[curr:curr+length] = id
            
    #         with autocast(device_type="cuda"):
    #             img = img.to(train_config.device)
    #             img_feature, img_map = extractor.extract(img)
                
    #         # save features in fp32 for sim calculation
    #         # img_features_list.append(img_feature.to(torch.float32))
    #         img_features[curr:curr+length] = img_feature.to(torch.float32)
    #         curr += length
    
    shape = (all, 1024, train_config.img_size//32, train_config.img_size//32)
    feature_cache = np.memmap(train_config.feature_cache, dtype="float32", mode="w+", shape=shape) if memmap else None
    
    for img, id in bar:
        # ids_list.append(id)
        length = len(id)
        ids[curr:curr+length] = id
        
        with autocast(device_type="cuda"):
            img = img.to(train_config.device)
            img_feature, img_map = extractor.extract(img)
            
            if memmap:
                feature_cache[curr:curr+length] = img_map.to(torch.float32).cpu().numpy()
        
            # # normalize is calculated in fp32
            # if train_config.normalize_features:
            #     img_feature = F.normalize(img_feature, dim=-1)
        
        # save features in fp32 for sim calculation
        # img_features_list.append(img_feature.to(torch.float32))
        img_features[curr:curr+length] = img_feature.to(torch.float32)
        
        curr += length
      
        # keep Features on GPU
        # img_features = torch.cat(img_features_list, dim=0) 
        # ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        ids = ids.to(train_config.device)
    ids = ids.to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return img_features, ids