import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from .models.CLIPScore import CLIPScore
from .models.BLIPScore import BLIPScore
from .models.AestheticScore import AestheticScore
from ImageFlow.ImageFlow import ImageFlow


def load(
    name: str = "ImageFLow-v1.0",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    download_root: str = None,
    med_config: str = "/data/wzh/aigc_ws/Instruct-PG/ImageFlow/med_config.json",
):
    """Load a ImageFLow model

    Parameters
    ----------
    name : str
        A model name listed by `ImageFLow.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    download_root: str
        path to download the model files; by default, it uses "~/.cache/ImageFLow"

    Returns
    -------
    model : torch.nn.Module
        The ImageFLow model
    """
    if os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found")

    print("load checkpoint from %s" % model_path)
    state_dict = torch.load(model_path, map_location="cpu")

    model = ImageFlow(device=device, med_config=med_config).to(device)
    msg = model.load_state_dict(state_dict, strict=False)
    print("checkpoint loaded")
    model.eval()

    return model
