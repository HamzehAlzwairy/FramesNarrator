#@title Imports

import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import skimage.io as io
import PIL.Image
from IPython.display import Image
from model import ClipCaptionModel
import generate
import argparse


N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]
D = torch.device
CPU = torch.device('cpu')

def get_device(device_id: int) -> D:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')

CUDA = get_device


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--image_path', type=str, default= 'images/image1.png' )
    args = parser.parse_args()

    current_directory = os.path.dirname(__file__)
    save_path = os.path.join(current_directory, "pretrained_models")
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, 'conceptual_weights.pt')
    # model_path = os.path.join(save_path, 'coco_weights.pt')

    is_gpu = True
    device = CUDA(0) if is_gpu else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    prefix_length = 10
    model = ClipCaptionModel(prefix_length)
    model.load_state_dict(torch.load(model_path, map_location=CPU))
    model = model.eval()
    device = CUDA(0) if is_gpu else "cpu"
    model = model.to(device)

    use_beam_search = False #@param {type:"boolean"}
    image = io.imread(args.image_path)
    pil_image = PIL.Image.fromarray(image)

    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        # if type(model) is ClipCaptionE2E:
        #     prefix_embed = model.forward_image(image)
        # else:
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
    if use_beam_search:
        generated_text_prefix = generate.generate_beam(model, tokenizer, embed=prefix_embed)[0]
    else:
        generated_text_prefix = generate.generate2(model, tokenizer, embed=prefix_embed)

    print('\n')
    print(generated_text_prefix)