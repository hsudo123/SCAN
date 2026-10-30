import os
import io
import cv2
import time
import copy
import lmdb
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from ..mercator import get_pixel_tensor

class VigorDatasetVirtual(Dataset):
    def __init__(
        self,
        data_folder,
        same_area=True,
        split="train",
        img_size=384
    ):
        super().__init__()
        self.image_size_sat = (img_size, img_size)
        self.img_size_ground = (img_size, 2*img_size)
        
    @staticmethod
    def get_meta(data_folder, same_area, split="train"):
        if same_area:
            cities = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        elif split == "train":
            cities = ['NewYork', 'Seattle']
        else:
            cities = ['SanFrancisco', 'Chicago']
            
        # load sat list 
        sat_list = []
        for city in cities:
            df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/satellite_list.txt', header=None, sep='\s+')
            df_tmp = df_tmp.rename(columns={0: "sat"})
            df_tmp["path"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/satellite/{x.sat}', axis=1)
            sat_list.append(df_tmp)
        df_sat = pd.concat(sat_list, axis=0).reset_index(drop=True)
        
        # idx for complete train and test independent of mode = train or test
        sat2idx = dict(zip(df_sat.sat, df_sat.index))
        # idx2sat = dict(zip(df_sat.index, df_sat.sat))
        # idx2sat_path = dict(zip(df_sat.index, df_sat.path))
        
        # ground dependent on mode 'train' or 'test'
        ground_list = []
        for city in cities:
            if same_area:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/same_area_balanced_{split}.txt', header=None, sep='\s+')
            else:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/pano_label_balanced.txt', header=None, sep='\s+')
                
            df_tmp = df_tmp.loc[:, [0, 1, 4, 7, 10]].rename(columns={0:  "ground",
                                                                     1:  "sat",
                                                                     4:  "sat_np1",
                                                                     7:  "sat_np2",
                                                                     10: "sat_np3"})
            df_tmp["path_ground"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/panorama/{x.ground}', axis=1)
            df_tmp["path_sat"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/satellite/{x.sat}', axis=1)
            
            for sat_n in ["sat", "sat_np1", "sat_np2", "sat_np3"]:
                df_tmp[f'{sat_n}'] = df_tmp[f'{sat_n}'].map(sat2idx)
            ground_list.append(df_tmp) 
        df_ground = pd.concat(ground_list, axis=0).reset_index(drop=True)
            
        # idx for split train or test dependent on mode = train or test
        # idx2ground = dict(zip(df_ground.index, df_ground.ground))
        # idx2ground_path = dict(zip(df_ground.index, df_ground.path_ground))
            
        return df_ground, df_sat
    
    def encode_img(self, path, size):
        with Image.open(path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            img = img.resize(size, Image.Resampling.LANCZOS)  # (W, H)
            buffer = io.BytesIO()
            # 使用 PNG 格式進行無損儲存
            img.save(buffer, format="PNG")

            # 從緩衝區取得位元組
            return buffer.getvalue()

    def decode_img(self, img_bytes):
        buffer = io.BytesIO(img_bytes)
        img = Image.open(buffer)
        return img
    
    def create_lmdb(
        self, 
        lmdb_path       : str, 
        df_ground       : pd.DataFrame, 
        df_sat          : pd.DataFrame,
        img_size_ground : tuple[int, int],
        image_size_sat  : tuple[int, int]
    ):
        map_size = 100 * 1024 * 1024 * 1024
        env = lmdb.open(lmdb_path, map_size=map_size)
        
        txn = env.begin(write=True)
        for i, path in enumerate(tqdm(df_ground.path_ground, desc="Save grd imgs")):
            txn.put(f"grd_img_{i}".encode(), self.encode_img(path, img_size_ground))
            
            grd_gps = np.array(path[:-5].split(',')[-2:]).astype(float)
            txn.put(f"grd_gps_{i}".encode(), grd_gps.tobytes())
            
            if (i + 1) % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
        txn = env.begin(write=True)
        
        for i, path in enumerate(tqdm(df_sat.path, desc="Save sat imgs")):
            txn.put(f"sat_img_{i}".encode(), self.encode_img(path, image_size_sat))
            
            sat_gps = np.array(path[:-4].split('_')[-2:]).astype(float)
            txn.put(f"sat_gps_{i}".encode(), sat_gps.tobytes())
            
            if (i + 1) % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
        txn = env.begin(write=True)
        

class VigorDatasetTrain(VigorDatasetVirtual):
    def __init__(
        self,
        data_folder,
        same_area=True,
        transforms_query=None,
        transforms_reference=None,
        prob_flip=0.0,
        prob_rotate=0.0,
        shuffle_batch_size=128,
        img_size=384
    ):
        super().__init__(data_folder, same_area, "train")

        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        
        self.transforms_query = transforms_query           # ground
        self.transforms_reference = transforms_reference   # satellite
        
        df_ground, df_sat = self.get_meta(data_folder, same_area, "train")
        
        # idx for complete train and test independent of mode = train or test
        sat2idx = dict(zip(df_sat.sat, df_sat.index))
        self.idx2sat = dict(zip(df_sat.index, df_sat.sat))
        self.idx2sat_path = dict(zip(df_sat.index, df_sat.path))
        
        # idx for split train or test dependent on mode = train or test
        self.idx2ground = dict(zip(df_ground.index, df_ground.ground))
        self.idx2ground_path = dict(zip(df_ground.index, df_ground.path_ground))

        self.pairs = list(zip(df_ground.index, df_ground.sat))
        self.idx2pairs = defaultdict(list)
        
        # for a unique sat_id we can have 1 or 2 ground views as gt
        for pair in self.pairs:      
            self.idx2pairs[pair[1]].append(pair)
            
        self.label = df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values 
        
        self.samples = copy.deepcopy(self.pairs)
        
        lmdb_path = os.path.join(data_folder, f"lmdb/{'same' if same_area else 'cross'}_train.lmdb")
        if not os.path.exists(lmdb_path):
            os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
            self.create_lmdb(lmdb_path, df_ground, df_sat, (2*img_size, img_size), (img_size, img_size))
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
            
    def __getitem__(self, index):
        # return query_img, reference_img, label
        idx_ground, idx_sat = self.samples[index]
        
        # load query -> ground image
        grd_path = self.idx2ground_path[idx_ground]
        grd_gps = torch.from_numpy(np.array(grd_path[:-5].split(',')[-2:]).astype(float))
        grd_img = cv2.imread(grd_path)
        grd_img = cv2.cvtColor(grd_img, cv2.COLOR_BGR2RGB)
        
        # load reference -> satellite image
        sat_path = self.idx2sat_path[idx_sat]
        sat_gps = torch.from_numpy(np.array(sat_path[:-4].split('_')[-2:]).astype(float))
        sat_img = cv2.imread(sat_path)
        sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)
        
        gt_h, gt_w = get_pixel_tensor(sat_gps[0], sat_gps[1], grd_gps[0], grd_gps[1], 20)
        # gt_h, gt_w = 2 * gt_h / self.sat_size - 1, 2 * gt_w / self.sat_size - 1
        gt_index = 2 * torch.stack([gt_h, gt_w]) / 640 - 1
            
        # Flip simultaneously query and reference
        if np.random.random() < self.prob_flip:
            grd_img = cv2.flip(grd_img, 1)
            sat_img = cv2.flip(sat_img, 1)
            gt_index[1] = -gt_index[1]
        
        # image transforms
        grd_img = self.transforms_query(image=grd_img)['image']
        sat_img = self.transforms_reference(image=sat_img)['image']
        # print(grd_img.shape, sat_img.shape)
                
        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1, 2, 3])
            
            # rotate sat img 90 or 180 or 270
            sat_img = torch.rot90(sat_img, k=r, dims=(1, 2)) 
            
            # use roll for ground view if rotate sat view
            c, h, w = grd_img.shape
            shifts = -w // 4 * r
            grd_img = torch.roll(grd_img, shifts=shifts, dims=2)
            
            gt_h, gt_w = gt_index
            if r == 1:
                gt_index = torch.stack([-gt_w, gt_h])
            elif r == 2:
                gt_index = torch.stack([-gt_h, -gt_w])
            elif r == 3:
                gt_index = torch.stack([gt_w, -gt_h])
        
        return grd_img, sat_img, gt_index
    
    def __len__(self):
        return len(self.samples)
               
    def shuffle(self, sim_dict=None, neighbour_select=8, neighbour_range=16):

            '''
            custom shuffle function for unique class_id sampling in batch
            '''
            
            print("\nShuffle Dataset:")
            
            pair_pool = copy.deepcopy(self.pairs)
            idx2pair_pool = copy.deepcopy(self.idx2pairs)
            
            neighbour_split = neighbour_select // 2
                        
            if sim_dict is not None:
                similarity_pool = copy.deepcopy(sim_dict)                
            
            # Shuffle pairs order
            random.shuffle(pair_pool)
           
            
            # Lookup if already used in epoch
            pairs_epoch = set()   
            idx_batch = set()
     
            
            # buckets
            batches = []
            current_batch = []
            
            
            # counter
            break_counter = 0
            
            # progressbar
            pbar = tqdm()
    
            while True:
                
                pbar.update()
                
                if len(pair_pool) > 0:
                    pair = pair_pool.pop(0)
                    
                    _, idx = pair
                    
                    if idx not in idx_batch and pair not in pairs_epoch and len(current_batch) < self.shuffle_batch_size:
                        
                        idx_batch.add(idx)
                        current_batch.append(pair)
                        pairs_epoch.add(pair)
                        
                        # remove from pool used for sim-sampling
                        idx2pair_pool[idx].remove(pair)

                        if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                            
                            near_similarity = copy.deepcopy(similarity_pool[idx][:neighbour_range])
                            near_always = copy.deepcopy(near_similarity[:neighbour_split]) 
                            near_random = copy.deepcopy(near_similarity[neighbour_split:])
                            random.shuffle(near_random)
                            near_random = near_random[:neighbour_split]
                            near_similarity_select = near_always + near_random

                            
                            for idx_near in near_similarity_select:
                            
                            
                                # check for space in batch
                                if len(current_batch) >= self.shuffle_batch_size:
                                    break
                            
                                # no check for pair in epoch necessary cause all we add is removed from pool
                                if idx_near not in idx_batch:
                            
                                    near_pairs = copy.deepcopy(idx2pair_pool[idx_near])
                                    
                                    # up to 2 for one sat view 
                                    random.shuffle(near_pairs)
                                
                                    for near_pair in near_pairs:
                                                                                    
                                        idx_batch.add(idx_near)
                                        current_batch.append(near_pair)
                                        pairs_epoch.add(near_pair)
                                        
                                        idx2pair_pool[idx_near].remove(near_pair)
                                        similarity_pool[idx].remove(idx_near)
                                        
                                        # only select one view
                                        break
                             
                        break_counter = 0
                        
                    else:
                        # if pair fits not in batch and is not already used in epoch -> back to pool
                        if pair not in pairs_epoch:
                            pair_pool.append(pair)
                            
                        break_counter += 1
                        
                    if break_counter >= 1024:
                        break
                   
                else:
                    break

                if len(current_batch) >= self.shuffle_batch_size:
                
                    # empty current_batch bucket to batches
                    batches.extend(current_batch)
                    idx_batch = set()
                    current_batch = []
       
            pbar.close()
            
            # wait before closing progress bar
            time.sleep(0.3)
            
            self.samples = batches
            print("pair_pool:", len(pair_pool))
            print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples))) 
            print("Break Counter:", break_counter)
            print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][1], self.samples[-1][1]))  

             
class VigorDatasetEval(VigorDatasetVirtual):
    def __init__(
        self,
        data_folder,
        split,
        img_type,
        same_area=True,
        transforms=None,
        img_size=384
    ):
        super().__init__(data_folder, same_area, split)

        self.split = split
        self.img_type = img_type
        self.transforms = transforms
        
        df_ground, df_sat = self.get_meta(data_folder, same_area, split)
        
        # idx for complete train and test independent of mode = train or test
        sat2idx = dict(zip(df_sat.sat, df_sat.index))
        self.idx2sat = dict(zip(df_sat.index, df_sat.sat))
        self.idx2sat_path = dict(zip(df_sat.index, df_sat.path))
        
        # idx for split train or test dependent on mode = train or test
        self.idx2ground = dict(zip(df_ground.index, df_ground.ground))
        self.idx2ground_path = dict(zip(df_ground.index, df_ground.path_ground))
        
        lmdb_path = os.path.join(data_folder, f"lmdb/{'same' if same_area else 'cross'}_{split if split != 'train' else 'train'}.lmdb")
        if not os.path.exists(lmdb_path):
            os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
            self.create_lmdb(lmdb_path, df_ground, df_sat, (2*img_size, img_size), (img_size, img_size))
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
        if self.img_type == "reference":
            if split == "train":
                # only sat images we really train on
                self.label = df_ground["sat"].unique()
                self.images = []
                for idx in self.label:
                    self.images.append(self.idx2sat_path[idx])
            else:
                # all sat images of cities in split
                self.images = df_sat["path"].values
                self.label = df_sat.index.values
            
        elif self.img_type == "query":
            self.images = df_ground["path_ground"].values
            self.label = df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values 

        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")

    def __getitem__(self, index):
        
        img_path = self.images[index]
        label = self.label[index]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
            
        label = torch.tensor(label, dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.images)