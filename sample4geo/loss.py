import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn

class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device

    def forward(self, grd_pano_feature, sat_pano_feature, grd_bev_feature, sat_bev_feature, attention_weight, gt_index, logit_scale):
        B, _, H, W = attention_weight.shape
        labels = torch.arange(B, dtype=torch.long, device=self.device)
        # pos = torch.eye(B, dtype=torch.bool, device=self.device)
        grd_pano_feature = F.normalize(grd_pano_feature, dim=-1)
        
        # retreival_matrix_1 = logit_scale[0] * grd_retrieval @ sat_retrieval.T
        pano_scan_matrix_1 = logit_scale[0] * torch.einsum(
            "GD,GSD->GS", 
            grd_pano_feature, 
            F.normalize(sat_pano_feature, dim=-1)
        )
        pano_scan_matrix_2 = pano_scan_matrix_1.T
        retreival_loss = (self.loss_function(pano_scan_matrix_1, labels) + self.loss_function(pano_scan_matrix_2, labels)) / 2

        if grd_bev_feature is not None:
            grd_bev_feature = F.normalize(grd_bev_feature, dim=-1)
            pano_bev_matrix_1 = logit_scale[1] * (grd_pano_feature @ grd_bev_feature.T)
            pano_bev_matrix_2 = pano_bev_matrix_1.T
            pano_bev_loss = (self.loss_function(pano_bev_matrix_1, labels) + self.loss_function(pano_bev_matrix_2, labels)) / 2
            
            bev_scan_matrix_1 = logit_scale[2] * torch.einsum(
                "GD,GSD->GS", 
                grd_bev_feature, 
                F.normalize(sat_bev_feature, dim=-1)
            )
            bev_scan_matrix_2 = bev_scan_matrix_1.T
            bev_scan_loss = (self.loss_function(bev_scan_matrix_1, labels) + self.loss_function(bev_scan_matrix_2, labels)) / 2
        else:
            pano_bev_loss, bev_scan_loss = None, None
        
        return retreival_loss, pano_bev_loss, bev_scan_loss
        # return retreival_loss, None, None
        
        # reranking_matrix_1 = logit_scale[1] * torch.einsum(
        #     "GD,GSD->GS", 
        #     grd_retrieval, 
        #     F.normalize(sat_reranking, dim=-1)
        # )
        # reranking_matrix_2 = reranking_matrix_1.T
        # reranking_loss = (self.loss_function(reranking_matrix_1, labels) + self.loss_function(reranking_matrix_2, labels)) / 2
        
        # position_loss = -torch.mean(torch.log(F.grid_sample(
        #     attention_weight[pos].unsqueeze(1),     # (B, 1, H, W)
        #     gt_index.unsqueeze(1).unsqueeze(1),  # (B, 1, 1, 2)
        #     align_corners=True
        # ).squeeze() + 1e-6))

        # return retreival_loss, reranking_loss, position_loss
 

