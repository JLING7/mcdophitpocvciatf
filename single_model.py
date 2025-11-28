import torch
import torch.nn as nn
import torchvision.models as models
from timm.layers import LayerNorm2d
import timm
from MDST_model import MDST

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, not isinstance(module, LayerNorm2d))
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output



class MDST_EfficientNet(nn.Module):
    def __init__(self, num_classes=4):
        super(MDST_EfficientNet, self).__init__()

        # Swin-Tiny backbone
        self.swin = MDST(
                pretrain_img_size=224,
                patch_size=4,
                in_chans=3,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.2,
                norm_layer=torch.nn.LayerNorm,
                ape=False,
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                frozen_stages=4,
                use_checkpoint=False,
                num_classes=4
            )
        self.swin.init_weights(pretrained='pretrained weights')
        swin_out = self.swin.num_features[-1]
        
        self.eff = models.efficientnet_b0(pretrained=True)
        #for param in self.eff.parameters():
        #    param.requires_grad = False
        eff_out = self.eff.classifier[1].in_features
        self.eff.classifier = nn.Identity() 

        # 融合后的分类器
        self.classifier = nn.Sequential(
            nn.Linear(swin_out + eff_out, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        f1 = self.swin(x)   
        f2 = self.eff(x)    
        f = torch.cat([f1, f2], dim=1)  
        out = self.classifier(f)
        return out


    

