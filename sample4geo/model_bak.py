import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import timm
from .vpr import MixVPR, SALAD

# class TimmModel(nn.Module):
#     def __init__(self, 
#                  model_name,
#                  pretrained=True,
#                  img_size=383):
                 
#         super(TimmModel, self).__init__()
#         self.img_size = img_size
        
#         if "vit" in model_name:
#             # automatically change interpolate pos-encoding to img_size
#             self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
#         else:
#             self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
#         self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
#     def get_config(self,):
#         data_config = timm.data.resolve_model_data_config(self.model)
#         return data_config
    
#     def set_grad_checkpointing(self, enable=True):
#         self.model.set_grad_checkpointing(enable)
        
#     def forward(self, img1, img2=None):
#         if img2 is not None:
#             image_features1 = self.model(img1)     
#             image_features2 = self.model(img2)
#             return image_features1, image_features2            
#         else:
#             image_features = self.model(img1)
#             return image_features


# class Extractor(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.model = timm.create_model(config.model, pretrained=True, features_only=True)
#         self.num_layers = config.num_layers
        
#         hidden_dim = config.hidden_dim
#         conv_list = []
#         dim, k = 128 * (2 ** (4 - self.num_layers)), 8 // (2 ** (4 - self.num_layers))
#         for i in range(self.num_layers):
#             conv_list.append(nn.Conv2d(
#                 in_channels=dim, 
#                 out_channels=hidden_dim//self.num_layers,
#                 kernel_size=k,
#                 stride=k
#             ))
#             dim, k = dim * 2, k // 2
#         self.conv_list = nn.ModuleList(conv_list)
#         self.init_weights(self.conv_list)
    
#     @staticmethod
#     def init_weights(module: nn.Module):
#         for m in module.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
        
#     def get_feature_map(self, img: torch.Tensor):
#         stages = self.model(img)[-self.num_layers:]
#         feature_map = torch.cat([conv(stage) for conv, stage in zip(self.conv_list, stages)], dim=1)
#         return feature_map
        
#     def forward(self, img: torch.Tensor):
#         raise NotImplementedError()

class GroundExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = timm.create_model(config.model, pretrained=True, num_classes=0)
        # self.aggregator = MixVPR()
        self.aggregator = SALAD(config)
    
    def forward(self, img: torch.Tensor):
        feature_map = self.model.forward_features(img)
        # global_feature = self.model.forward_head(feature_map, pre_logits=True)
        global_feature = self.aggregator(feature_map)
        return global_feature
        
        
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
        hidden_dim = config.hidden_dim
        latent_dim = config.latent_dim
        self.sqrt_dim = hidden_dim ** 0.5
        self.grd_normalization = nn.LayerNorm(latent_dim, eps=1e-12)
        self.sat_normalization = timm.layers.LayerNorm2d(1024)
        
        # self.sat_conv = nn.Sequential(
        #     nn.Conv2d(1024, 1024, kernel_size=3, padding=1, groups=1024, bias=False),   # DW
        #     nn.GELU(),
        #     nn.Conv2d(1024, 1024, kernel_size=1, bias=False)                            # PW
        # )
        
        self.query_proj = nn.Linear(latent_dim, hidden_dim)
        self.key_proj   = nn.Conv1d(in_channels=1024, out_channels=hidden_dim, kernel_size=1, bias=False)
        # self.value_proj = nn.Conv1d(in_channels=1024, out_channels=hidden_dim, kernel_size=1)
        self.value_proj = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
        )
        
        self.out        = nn.Linear(hidden_dim, latent_dim)
        
        # model = timm.create_model(config.model, pretrained=True, num_classes=0)
        # self.head = model.head
        
        self.dropout = nn.Dropout(p=0.1)
        
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
        # global_feature = self.head(sat_map).unsqueeze(0).expand((B_grd, -1, -1))
        
        grd_feature = self.grd_normalization(grd_feature)               # (B_grd, dim)
        # sat_map = self.sat_normalization(sat_map).flatten(start_dim=2)  # (B_sat, 1024, h * w)
        sat_map = self.sat_normalization(sat_map)                       # (B_sat, 1024, h, w)
        
        # sat_map = self.sat_conv(sat_map)
        sat_map = sat_map.flatten(start_dim=2)
        
        query = self.query_proj(grd_feature)                            # (B_grd, dim)
        key = self.key_proj(sat_map)                                    # (B_sat, dim, h * w)
        value = self.value_proj(sat_map)                                # (B_sat, dim, h * w)
        # key = self.dropout(self.key_proj(sat_map))                                    # (B_sat, dim, h * w)
        # value = self.dropout(self.value_proj(sat_map))                                # (B_sat, dim, h * w)
        
        weight = torch.softmax(torch.einsum(                            # (B_grd, B_sat, h * w)
            "GD,SDL->GSL", 
            query, 
            key
        ) / self.sqrt_dim, dim=-1)
        
        aggregate = self.out(torch.einsum(                              # (B_grd, B_sat, dim)       
            "GSL,SDL->GSD", 
            weight, 
            value
        ))
        return weight.reshape(B_grd, B_sat, h, w), aggregate
        # return weight.reshape(B_grd, B_sat, h, w), self.dropout(aggregate)


class TimmModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.grd_extractor = GroundExtractor(config)
        self.sat_extractor = SatelliteExtractor(config)
        self.attention = ScanAttention(config) if config.true_attention else FakeScanAttention(config)
        
        self.temperature = nn.Parameter(torch.Tensor([np.log(1 / 0.07), np.log(1 / 0.07)])) # 1 / 0.07
        # self.temperature = nn.Parameter(torch.Tensor([np.log(10), np.log(10)])) # 1 / 0.07
    
    def set_grad_checkpointing(self, mode: bool=False):
        self.grd_extractor.model.set_grad_checkpointing(mode)
        self.sat_extractor.model.set_grad_checkpointing(mode)
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.grd_extractor)
        return data_config
    
    def forward(self, grd: torch.Tensor, sat: torch.Tensor):
        grd_retrieval = self.grd_extractor(grd)
        sat_map = self.sat_extractor(sat)
        
        attention_weight, sat_reranking = self.attention(grd_retrieval, sat_map)
        
        return grd_retrieval, None, sat_reranking, attention_weight
        # return grd_retrieval, sat_retrieval, sat_reranking, attention_weight
        
        
class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 in_h=12,
                 in_w=24,
                 out_channels=1024,
                 mix_depth=4,
                 mlp_ratio=1,
                 out_rows=1,
                 ) -> None:
        super().__init__()

        self.in_h = in_h # height of input feature maps
        self.in_w = in_w # width of input feature maps
        self.in_channels = in_channels # depth of input feature maps
        
        self.out_channels = out_channels # depth wise projection dimension
        self.out_rows = out_rows # row wise projection dimesion

        self.mix_depth = mix_depth # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio # ratio of the mid projection layer in the mixer block

        hw = in_h*in_w
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        # x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x.flatten(1)

# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        M : torch.Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)

# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
def get_matching_probs(S, dustbin_score = 1.0, num_iters=3, reg=1.0):
    """sinkhorn"""
    batch_size, m, n = S.size()
    # augment scores matrix
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # prepare normalized source and target log-weights
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n-m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_P = log_otp_solver(
        log_a,
        log_b,
        S_aug,
        num_iters=num_iters,
        reg=reg
    )
    return log_P - norm

class SALAD(nn.Module):
    """
    This class represents the Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD) model.

    Attributes:
        num_channels (int): The number of channels of the inputs (d).
        num_clusters (int): The number of clusters in the model (m).
        cluster_dim (int): The number of channels of the clusters (l).
        token_dim (int): The dimension of the global scene token (g).
        dropout (float): The dropout rate.
    """
    def __init__(self,
            config
        ) -> None:
        super().__init__()
        num_channels = 1024
        self.num_clusters = config.num_clusters
        self.cluster_dim = config.cluster_dim
        self.sinkhorn_iter = config.sinkhorn_iter
        print(f"SALAD cluster num: {self.num_clusters}, dim: {self.cluster_dim}")
        
        dropout = nn.Dropout(0.3)
        # dropout = nn.Identity()

        # MLP for global scene token g
        # self.token_features = nn.Sequential(
        #     nn.Linear(self.num_channels, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, self.token_dim)
        # )
        
        # normalize input features
        # self.pre_norm = nn.LayerNorm(num_channels)
        
        # MLP for local features f_i
        self.cluster_features = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, groups=num_channels),  # DW conv
            nn.Conv2d(num_channels, 512, 1),
            dropout,
            nn.GELU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )
        
        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(num_channels, 512, 1),
            dropout,
            nn.GELU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        # self.pos_proj = nn.Conv2d(4, 16, kernel_size=1)
        
        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.0))
        
    # def get_pos_info(self, feature_map):
    #     B, C, h, w = feature_map.shape
    #     phi = torch.linspace(-math.pi/2, math.pi/2, h).to(feature_map)
    #     theta = torch.linspace(-math.pi, math.pi, w).to(feature_map)
    #     phi, theta = torch.meshgrid(phi, theta, indexing="ij")
    #     position = torch.stack([
    #         torch.sin(theta),
    #         torch.cos(theta),
    #         torch.sin(phi),
    #         torch.cos(phi)
    #     ], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        
    #     return self.pos_proj(position)

    def forward(self, feature_map):
        """
        x (torch.Tensor): The feature tensors (t_i) [B, C, H, W].

        Returns:
            f (torch.Tensor): The global descriptor [B, m*l]
        """
        
        # feature_map = self.pre_norm(feature_map.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # p = self.score(torch.concat([
        #     feature_map, 
        #     self.get_pos_info(feature_map)
        # ], dim=1)).flatten(2)                               # (B, num_clusters, h * w)
        f = self.cluster_features(feature_map).flatten(2)   # (B, cluster_dim, h * w)
        p = self.score(feature_map).flatten(2)              # (B, num_clusters, h * w)

        # Sinkhorn algorithm
        p = get_matching_probs(p, self.dust_bin, self.sinkhorn_iter)
        p = torch.exp(p)
        # Normalize to maintain mass
        p = p[:, :-1, :]
        
        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        feature = (f * p).sum(dim=-1).flatten(1)
        
        # return nn.functional.normalize(feature, dim=-1)
        return feature

        # f = torch.cat([
        #     nn.functional.normalize(t, p=2, dim=-1),
        #     nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
        # ], dim=-1)
        # feature = torch.einsum("BCN,BKN->BKC", f, p).flatten(1)