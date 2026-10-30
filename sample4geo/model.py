import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import timm
from .vpr import SALAD, FakeSALAD

class GroundExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = timm.create_model(config.model, pretrained=True, num_classes=0)
        # self.aggregator = MixVPR()
        # self.aggregator = SALAD(config)
    
    def forward(self, img: torch.Tensor):
        feature_map = self.model.forward_features(img)
        # global_feature = self.model.forward_head(feature_map, pre_logits=True)
        # global_feature = self.aggregator(feature_map)
        return feature_map
        
        
class SatelliteExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = timm.create_model(config.model, pretrained=True, num_classes=0)
        
    def forward(self, img: torch.Tensor):
        feature_map = self.model.forward_features(img)
        return feature_map
    
    
class FakeScanAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        model = timm.create_model(config.model, pretrained=True, num_classes=0)
        self.head = model.head
        
    def forward(self, grd_feature: torch.Tensor, sat_map: torch.Tensor) -> torch.Tensor:
        B_grd, _ = grd_feature.shape
        B_sat, _, h ,w = sat_map.shape
        
        feature = self.head(sat_map)    # (B_sat, 1024)
        return torch.empty(B_grd, B_sat, h, w).to(feature), feature.expand(B_grd, -1, -1)
    

class ScanAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # hidden_dim = config.hidden_dim
        hidden_dim = 1024
        # self.num_heads = config.num_heads
        latent_dim = config.latent_dim
        self.sqrt_dim = hidden_dim ** 0.5
        # self.sqrt_dim = (hidden_dim / self.num_heads) ** 0.5
        self.grd_normalization = nn.LayerNorm(latent_dim, eps=1e-12)
        # self.sat_normalization = timm.layers.LayerNorm2d(1024)
        self.sat_normalization = nn.LayerNorm(latent_dim, eps=1e-12)
        
        # self.query_proj = nn.Linear(latent_dim, hidden_dim)
        # self.key_proj   = nn.Conv1d(in_channels=1024, out_channels=hidden_dim, kernel_size=1, bias=False)
        # self.value_proj = nn.Conv1d(in_channels=1024, out_channels=hidden_dim, kernel_size=1)
        
        dropout = nn.Dropout(p=0.1)
        self.query_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            dropout,
            nn.Linear(latent_dim, hidden_dim)
        )
        self.key_proj = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.GELU(),
            dropout,
            nn.Conv1d(in_channels=1024, out_channels=hidden_dim, kernel_size=1)
        )
        self.value_proj = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.GELU(),
            dropout,
            nn.Conv1d(in_channels=1024, out_channels=hidden_dim, kernel_size=1)
        )
        
        # self.out        = nn.Linear(hidden_dim, latent_dim)
        self.out        = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            dropout
        )
        
        self.kv_init_scale = 0.5
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Conv1d):
                # 對 key/value 的 conv 使用 Kaiming，但 scale-down
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if hasattr(self, "kv_init_scale") and self.kv_init_scale is not None:
                    with torch.no_grad():
                        m.weight.mul_(self.kv_init_scale)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Conv2d):
                # depthwise 與 pointwise 務求穩定
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, (nn.LayerNorm, timm.layers.LayerNorm2d)):
                # LayerNorm weight/bias 初始為 1 / 0
                try:
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0)
                except Exception:
                    pass
        
    def forward(self, grd_feature: torch.Tensor, sat_map: torch.Tensor) -> torch.Tensor:
        '''
        grd_feature : (B_grd, dim)        as query
        sat_map     : (B_sat, dim, h, w)  as key, value
        '''
        B_grd, _ = grd_feature.shape
        B_sat, _, h ,w = sat_map.shape
        
        grd_ln = self.grd_normalization(grd_feature)                    # (B_grd, dim)
        # sat_map = self.sat_normalization(sat_map).flatten(start_dim=2)  # (B_sat, 1024, h * w)
        sat_ln = self.sat_normalization(
            sat_map.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2).flatten(start_dim=2)                      # (B_sat, 1024, h * w)
        sat_map = sat_map.flatten(start_dim=2)
        
        # query = self.query_proj(grd_ln)                            # (B_grd, dim)
        # key = self.key_proj(sat_ln)                                    # (B_sat, dim, h * w)
        # value = self.value_proj(sat_ln)                                # (B_sat, dim, h * w)
        query = self.query_proj(grd_ln) + grd_feature                   # (B_grd, dim)
        key = self.key_proj(sat_ln) + sat_map                           # (B_sat, dim, h * w)
        value = self.value_proj(sat_ln) + sat_map                       # (B_sat, dim, h * w)
        
        weight = torch.softmax(torch.einsum(                            # (B_grd, B_sat, h * w)
            "GD,SDL->GSL", 
            query, 
            key
        ).float() / self.sqrt_dim, dim=-1)
        
        aggregate = self.out(torch.einsum(                              # (B_grd, B_sat, dim)       
            "GSL,SDL->GSD", 
            weight, 
            value
        ))
        # aggregate = torch.einsum(                              # (B_grd, B_sat, dim)       
        #     "GSL,SDL->GSD", 
        #     weight, 
        #     value
        # )
        # aggregate = self.out(aggregate) + aggregate
        
        # weight = torch.softmax(torch.einsum(                            # (B_grd, B_sat, h * w)
        #     "GHD,SHDL->GSHL", 
        #     query.view(B_grd, self.num_heads, -1), 
        #     key.view(B_sat, self.num_heads, -1, h * w)
        # ).float() / self.sqrt_dim, dim=-1)
        
        # aggregate = self.out(torch.einsum(                              # (B_grd, B_sat, dim)       
        #     "GSHL,SHDL->GSHD", 
        #     weight, 
        #     value.view(B_sat, self.num_heads, -1, h * w)
        # ).flatten(start_dim=2))
        
        # weight = weight.mean(dim=2)
        return weight.reshape(B_grd, B_sat, h, w), aggregate
        # return weight.reshape(B_grd, B_sat, h, w), self.dropout(aggregate)


class SCAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.grd_extractor = GroundExtractor(config)
        self.pano_aggregator = SALAD(config)
        if config.use_bev:
            self.bev_extractor = SatelliteExtractor(config)
            self.bev_aggregator = SALAD(config)
        self.sat_extractor = SatelliteExtractor(config)
        self.attention = ScanAttention(config) if config.true_attention else FakeScanAttention(config)
        
        self.temperature = nn.Parameter(torch.Tensor([np.log(1 / 0.07), np.log(1 / 0.07), np.log(1 / 0.07)])) # 1 / 0.07
    
    def set_grad_checkpointing(self, mode: bool=False):
        self.grd_extractor.model.set_grad_checkpointing(mode)
        if hasattr(self, "bev_extractor"): 
            self.bev_extractor.model.set_grad_checkpointing(mode)
        self.sat_extractor.model.set_grad_checkpointing(mode)
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.grd_extractor)
        return data_config
    
    def forward(self, grd: torch.Tensor, sat: torch.Tensor):
        grd_feature = self.pano_aggregator(self.grd_extractor(grd))
        sat_map = self.sat_extractor(sat)
        
        attention_weight, sat_feature = self.attention(grd_feature, sat_map)
        
        return grd_feature, sat_feature, attention_weight
    
    def train_forward(self, grd_pano: torch.Tensor, grd_bev: torch.Tensor, sat: torch.Tensor):
        grd_pano_feature = self.pano_aggregator(self.grd_extractor(grd_pano))
        # grd_bev_feature = self.bev_aggregator(self.sat_extractor(grd_bev))
        grd_bev_feature = self.bev_aggregator(self.bev_extractor(grd_bev))
        sat_map = self.sat_extractor(sat)
        
        attention_weight, sat_pano_feature = self.attention(grd_pano_feature, sat_map)
        _, sat_bev_feature = self.attention(grd_bev_feature, sat_map)
        
        return grd_pano_feature, grd_bev_feature, sat_pano_feature, sat_bev_feature, attention_weight