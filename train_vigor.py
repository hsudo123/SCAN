import os
import shutil
import sys
import torch
import pickle
from dataclasses import dataclass
from torch.amp import GradScaler
# from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import \
    get_constant_schedule_with_warmup, \
    get_polynomial_decay_schedule_with_warmup, \
    get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup

from sample4geo.dataset.vigor import VigorDatasetEval, VigorDatasetTrain
from sample4geo.transforms import get_transforms_train, get_transforms_val
from sample4geo.utils import setup_system, Logger
from sample4geo.loss import InfoNCE
from sample4geo.model import SCAN
# from sample4geo.trainer import train
# from sample4geo.evaluate.vigor import evaluate, calc_sim
from sample4geo.new_trainer import train
from sample4geo.evaluate.new_vigor import evaluate, calc_sim

@dataclass
class Configuration:
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    true_attention: bool = True
    num_clusters: int = 64
    cluster_dim: int = 16
    latent_dim: int = num_clusters * cluster_dim
    hidden_dim: int = latent_dim * 2    # 1024 now
    sinkhorn_iter: int = 20
    num_heads: int = 8
    bev_coeff: float = 0.1
    
    # Override model image size
    img_size: int = 384
    sat_size: int = 640
    
    # Training 
    mixed_precision: bool = True
    seed = 1
    batch_size: int = 56        # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    use_bev: bool = True
    
    # Augment Images
    prob_rotate: float = 0.75          # rotates the sat image and ground images simultaneously
    prob_flip: float = 0.5             # flipping the sat image and ground images simultaneously
    
    # train on GPU if available
    gpu_ids: tuple = (4,)   # GPU ids for training
    device: str = f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu' 
    buffer_device = f'cuda:{gpu_ids[0]+1}' if torch.cuda.is_available() else 'cpu' 
    
    # Dataset
    data_folder = "/home/nvidia/.cache/VIGOR"
    same_area: bool = False             # True: same | False: cross
    ground_cutting = 0                 # cut ground upper and lower
    pre_resized = False
    
    # Similarity Sampling
    custom_sampling: bool = True   # use custom sampling instead of random
    gps_sample: bool = True        # use gps sampling
    sim_sample: bool = True        # use similarity sampling
    neighbour_select: int = batch_size // 2     # max selection size from pool
    neighbour_range: int = batch_size     # pool size for selection
    gps_dict_path: str = "./gps_dict_same.pkl" if same_area else "./gps_dict_cross.pkl" # gps_dict_cross.pkl | gps_dict_same.pkl
    
    # Eval
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 10      # eval every n Epoch
    sample_every_n_epoch: int = 4
    normalize_features: bool = True

    # Optimizer 
    clip_grad = 100.                 # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = True # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.05 if same_area else 0.1
    
    # Learning Rate
    epochs: int = 90
    lr: float = 2.0 * 1e-4                 # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "seq"          # "polynomial" | "cosine" | "constant" | "seq" | None
    warmup_epochs: int = 5
    lr_end: float = 1e-5            #  only for "polynomial"
    epoch_cycle: int = 40
    epoch_milestone: int = epochs - 40
    
    
    # Checkpoint to start from
    checkpoint_start = None
    # checkpoint_start = "/media/nvidia/E/mkrxp123/Sample4Geo/new_log/ablation_withoutSALAD/status_e10_49.4573.pth"
    # checkpoint_start = "/media/nvidia/E/mkrxp123/Sample4Geo/new_log/ablation_salad/status_e80_75.7799.pth"
    
    # Savepath for model checkpoints
    # model_path: str = f"./log/{'same' if same_area else 'cross'}_{num_clusters}*{cluster_dim}_{sinkhorn_iter}iter_{lr}_noshare"
    model_path: str = f"./new_log/ablation_cross_all"
    
    # Eval before training
    zero_shot: bool = False  
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4
    
    # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False
    
    # loss weight
    # reranking_coeff = 0.5
    # position_coeff = 0.1
    # temperature_coeff = 0.01


#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

config = Configuration() 


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([f"{i}" for i in [3, 4, 5]])


    # model_path = "{}/{}/{}".format(config.model_path,
    #                                    config.model,
    #                                    time.strftime("%H%M%S"))
    model_path = config.model_path

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config))


    # model = TimmModel(config.model,
    #                       pretrained=True,
    #                       img_size=config.img_size)

    model = SCAN(config)
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size
    
    image_size_sat = (img_size, img_size)

    new_width = img_size*2    
    new_hight = int(((1024 - 2 * config.ground_cutting) / 2048) * new_width)
    img_size_ground = (new_hight, new_width)
    
    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
     
    # # Load pretrained Checkpoint    
    # if config.checkpoint_start is not None:  
    #     print("Start from:", config.checkpoint_start)
    #     ckpt = torch.load(config.checkpoint_start, map_location="cpu")  
    #     model.load_state_dict(ckpt, strict=False)     

    # Data parallel
    # print("GPUs available:", torch.cuda.device_count())  
    # if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Model to device   
    model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    sat_transforms_train, ground_transforms_train = get_transforms_train(image_size_sat,
                                                                         img_size_ground,
                                                                         mean=mean,
                                                                         std=std,
                                                                         ground_cutting=config.ground_cutting,
                                                                         resized=config.pre_resized)
                                                                   
                                                                   
    # Train
    train_dataset = VigorDatasetTrain(data_folder=config.data_folder ,
                                      same_area=config.same_area,
                                      transforms_query=ground_transforms_train,
                                      transforms_reference=sat_transforms_train,
                                      prob_flip=config.prob_flip,
                                      prob_rotate=config.prob_rotate,
                                      shuffle_batch_size=config.batch_size
                                      )
    
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)
    
    
    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   ground_cutting=config.ground_cutting,
                                                                   resized=config.pre_resized)


    # Reference Satellite Images Test
    reference_dataset_test = VigorDatasetEval(data_folder=config.data_folder ,
                                              split="test",
                                              img_type="reference",
                                              same_area=config.same_area,  
                                              transforms=sat_transforms_val,
                                              )
    
    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size_eval,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)
    
    
    
    # Query Ground Images Test
    query_dataset_test = VigorDatasetEval(data_folder=config.data_folder ,
                                          split="test",
                                          img_type="query",
                                          same_area=config.same_area,      
                                          transforms=ground_transforms_val,
                                          )
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    
    print("Query Images Test:", len(query_dataset_test))
    print("Reference Images Test:", len(reference_dataset_test))
    

    #-----------------------------------------------------------------------------#
    # GPS Sample                                                                  #
    #-----------------------------------------------------------------------------#
    if config.gps_sample:
        print(config.gps_dict_path)
        with open(config.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
    else:
        sim_dict = None
    
    #-----------------------------------------------------------------------------#
    # Sim Sample + Eval on Train                                                  #
    #-----------------------------------------------------------------------------#
    
    if config.sim_sample:
        # Query Ground Images Train for simsampling
        query_dataset_train = VigorDatasetEval(data_folder=config.data_folder ,
                                               split="train",
                                               img_type="query",
                                               same_area=config.same_area,      
                                               transforms=ground_transforms_val,
                                               )
            
        query_dataloader_train = DataLoader(query_dataset_train,
                                            batch_size=config.batch_size_eval,
                                            num_workers=config.num_workers,
                                            shuffle=False,
                                            pin_memory=True)
        
        # Reference Satellite Images Train for simsampling
        reference_dataset_train = VigorDatasetEval(data_folder=config.data_folder ,
                                                   split="train",
                                                   img_type="reference",
                                                   same_area=config.same_area,  
                                                   transforms=sat_transforms_val,
                                                   )
        
        reference_dataloader_train = DataLoader(reference_dataset_train,
                                                batch_size=config.batch_size_eval,
                                                num_workers=config.num_workers,
                                                shuffle=False,
                                                pin_memory=True)
            
      
        print("\nQuery Images Train:", len(query_dataset_train))
        print("Reference Images Train (unique):", len(reference_dataset_train))
        
    
    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn,
                            device=config.device,
                            )

    if config.mixed_precision:
        scaler = GradScaler(device="cuda", init_scale=2.**10)
    else:
        scaler = None
        
    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        # optimizer_parameters = [
        #     {"params": model.sat_extractor.parameters(), "lr": config.lr * 0.1},
        #     {"params": model.attention.parameters(), "lr": config.lr},
        #     {"params": model.grd_extractor.model.parameters(), "lr": config.lr * 0.1},
        #     {"params": model.grd_extractor.aggregator.parameters(), "lr": config.lr},
        #     {"params": model.temperature, "lr": config.lr},
        # ]
        # optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
        # optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=0.05, adaptive=True, lr=config.lr)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)


    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    epoch_steps = len(train_dataloader)
    train_steps = epoch_steps * config.epochs
    warmup_steps = epoch_steps * config.warmup_epochs
       
    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))  
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_training_steps=train_steps,
            lr_end = config.lr_end,
            power=1.5,
            num_warmup_steps=warmup_steps
        )
    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))   
        scheduler =  get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps
        )
    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer=optimizer,
        #     T_max=train_steps,
        #     eta_min=config.lr_end
        # )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer=optimizer,
        #     T_0=epoch_steps * config.epoch_cycle,
        #     T_mult=1,
        #     eta_min=config.lr_end
        # )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=train_steps,
            num_warmup_steps=warmup_steps
        )
        # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        #     optimizer,
        #     num_training_steps=train_steps,
        #     num_warmup_steps=warmup_steps,
        #     num_cycles=config.epochs // config.epoch_cycle
        # )
    elif config.scheduler == "seq":
        # s1 = torch.optim.lr_scheduler.ConstantLR(
        #     optimizer=optimizer,
        #     factor=1,
        #     total_iters=epoch_steps * config.epoch_milestone
        # )
        s1 = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=1e-6,  # LR 的起始比例 (接近 0)
            end_factor=1.0,     # LR 的結束比例 (max_lr)
            total_iters=epoch_steps * config.warmup_epochs
        )
        s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=epoch_steps * (config.epochs - config.epoch_milestone),
            eta_min=0
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[s1, s2],
            milestones=[epoch_steps * config.epoch_milestone]
        )
    else:
        scheduler = None
        
    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))
        
        
    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))  

        r1_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=reference_dataloader_test,
                           query_dataloader=query_dataloader_test, 
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)
        
        if config.sim_sample:
            r1_train, sim_dict = calc_sim(config=config,
                                          model=model,
                                          reference_dataloader=reference_dataloader_train,
                                          query_dataloader=query_dataloader_train, 
                                          ranks=[1, 5, 10],
                                          step_size=1000,
                                          cleanup=True)
        
            
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    start_epoch = 0
    step = 0
    best_score = 0

    writer = SummaryWriter(model_path)
    # calc_sim(
    #     config=config,
    #     model=model,
    #     reference_dataloader=reference_dataloader_train,
    #     query_dataloader=query_dataloader_train, 
    #     ranks=[1, 5, 10],
    #     # step_size=1000,
    #     cleanup=True
    # )
    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        ckpt = torch.load(config.checkpoint_start, map_location="cpu")
        
        start_epoch = ckpt["epoch"]
        step = ckpt["step"]
        
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        sim_dict = ckpt["sim_dict"]
    # start_epoch = 0

    # print("eval")
    # r1_test = evaluate(
    #     config=config,
    #     model=model,
    #     reference_dataloader=reference_dataloader_test,
    #     query_dataloader=query_dataloader_test, 
    #     ranks=[1, 5, 10],
    #     # step_size=1000,
    #     cleanup=True
    # )
    # exit(0)
    
    #-----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    #-----------------------------------------------------------------------------#            
    # if config.custom_sampling:
    #     train_dataloader.dataset.shuffle(
    #         sim_dict,
    #         neighbour_select=config.neighbour_select,
    #         neighbour_range=config.neighbour_range
    #     )
    
    for epoch in range(start_epoch+1, config.epochs+1):
        if config.custom_sampling:
            train_dataloader.dataset.shuffle(
                sim_dict,
                neighbour_select=config.neighbour_select,
                neighbour_range=config.neighbour_range
            )
        
        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))
        train_loss = train(
            config,
            model,
            dataloader=train_dataloader,
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            writer=writer,
            curr_step=step
        )
        step += epoch_steps
        
        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))
        
        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))
        
            r1_test = evaluate(
                config=config,
                model=model,
                reference_dataloader=reference_dataloader_test,
                query_dataloader=query_dataloader_test, 
                ranks=[1, 5, 10],
                # step_size=1000,
                cleanup=True
            )
            
            if r1_test > best_score:
                best_score = r1_test

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    # torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                    weight = model.module.state_dict()
                else:
                    # torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                    weight = model.state_dict()
                    
                status = {
                    "epoch": epoch,
                    "step": step,
                    "model": weight,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "sim_dict": sim_dict
                }
                torch.save(status, '{}/status_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))

        if (epoch % config.sample_every_n_epoch == 0 and epoch != 0 and epoch != config.epochs) and config.sim_sample:
            r1_train, sim_dict = calc_sim(
                config=config,
                model=model,
                reference_dataloader=reference_dataloader_train,
                query_dataloader=query_dataloader_train, 
                ranks=[1, 5, 10],
                # step_size=1000,
                cleanup=True
            )
                
    # if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
    #     # torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    #     weight = model.module.state_dict()
    # else:
    #     # torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))
    #     weight = model.state_dict()
    #     status = {
    #         "epoch": config.epochs,
    #         "step": step,
    #         "model": weight,
    #         "optimizer": optimizer.state_dict(),
    #         "scheduler": scheduler.state_dict(),
    #         "scaler": scaler.state_dict(),
    #         "sim_dict": sim_dict
    #     }
    #     torch.save(status, '{}/status_end.pth'.format(model_path))