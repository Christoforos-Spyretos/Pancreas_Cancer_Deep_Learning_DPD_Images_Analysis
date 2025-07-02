# %%
import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
from HIPT_4K.hipt_model_utils import get_vit256
from models.open_clip_custom import create_model_from_pretrained
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'conch_v1':
        model, _ = create_model_from_pretrained("conch_ViT-B-16", '/local/data1/chrsp39/Pancreas_Cancer_Deep_Learning_DPD_Images_Analysis/models/CONCH/checkpoints/conch/conch_v1_pytorch_model.bin')
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
        model = model.to(device)
    elif model_name == 'conch_v1_5':
        model, _ = create_model_from_pretrained("conch_v1_5_official", '/local/data1/chrsp39/Pancreas_Cancer_Deep_Learning_DPD_Images_Analysis/models/CONCHV1_5/checkpoints/conch_v1_5/conch_v1_5pytorch_model.bin')
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
        model = model.to(device)
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)

    return model, img_transforms