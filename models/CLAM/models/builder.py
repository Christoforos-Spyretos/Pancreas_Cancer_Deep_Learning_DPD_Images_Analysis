# %%
import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
from models.open_clip_custom import create_model_from_pretrained
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'conch_v1':
        model, _ = create_model_from_pretrained("conch_ViT-B-16", '/local/data1/chrsp39/Pancreas_Cancer_Deep_Learning_DPD_Images_Analysis/models/CONCH_v1/checkpoints/conch/pytorch_model.bin')
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
        model = model.to(device)
    elif model_name == 'conch_v1_5':
        model, _ = create_model_from_pretrained("conch_v1_5_official", '/local/data1/chrsp39/Pancreas_Cancer_Deep_Learning_DPD_Images_Analysis/models/CONCH_v1_5/checkpoints/conch_v1_5/pytorch_model.bin')
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
        model = model.to(device)
    elif model_name == 'uni2-h':
        timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
            }
        model = timm.create_model(pretrained=False, **timm_kwargs)
        model.load_state_dict(torch.load(os.path.join("/local/data1/chrsp39/Pancreas_Cancer_Deep_Learning_DPD_Images_Analysis/models/UNI2-h/checkpoints/uni2-h", "pytorch_model.bin"), map_location="cpu"), strict=True)
        model = model.to(device)
    elif model_name == 'uni':
        model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
        model.load_state_dict(torch.load("/local/data1/chrsp39/Pancreas_Cancer_Deep_Learning_DPD_Images_Analysis/models/UNI/checkpoints/uni/pytorch_model.bin", map_location=device), strict=True)
        model = model.to(device)
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)

    return model, img_transforms