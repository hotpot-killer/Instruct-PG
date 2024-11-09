import os
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from .models.BLIP.blip_pretrain import BLIP_Pretrain

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def convert_to_rgb(img):
    return img.convert("RGB")


def preprocess_transform(size):
    return Compose(
        [
            Resize(size, interpolation=BICUBIC),
            CenterCrop(size),
            convert_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # Initialize MLP parameters
        for name, param in self.layers.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (self.input_dim + 1))
            if "bias" in name:
                nn.init.constant_(param, val=0)

    def forward(self, x):
        return self.layers(x)


class VisualTextModel(nn.Module):
    def __init__(self, config, device="cpu"):
        super().__init__()
        self.device = device

        self.blip_model = BLIP_Pretrain(image_size=224, vit="large", med_config=config)
        self.preprocess = preprocess_transform(224)
        self.mlp = MultiLayerPerceptron(768)

        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072

    def compute_score(self, prompt, img):

        # Text encoding
        text_input = self.blip_model.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)

        # Image encoding
        if isinstance(img, Image.Image):
            pil_img = img
        elif isinstance(img, str):
            if os.path.isfile(img):
                pil_img = Image.open(img)
        else:
            raise TypeError(
                "This image parameter type is not supported. Please pass a PIL.Image or file path string."
            )

        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        img_embeds = self.blip_model.visual_encoder(img_tensor)

        # Cross-attention between text and image
        img_atts = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(self.device)
        text_output = self.blip_model.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=img_embeds,
            encoder_attention_mask=img_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[:, 0, :].float()
        scores = self.mlp(txt_features)
        normalized_scores = (scores - self.mean) / self.std

        return normalized_scores.detach().cpu().numpy().item()

    def compute_scores(self, prompt, img):

        if type(img).__name__ == "list":
            _, scores = self.rank_inference(prompt, img)
            return scores
        text_input = self.blip_model.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)

        if isinstance(img, Image.Image):
            pil_img = img
        elif isinstance(img, str):
            if os.path.isfile(img):
                pil_img = Image.open(img)
        else:
            raise TypeError(
                "This image parameter type is not supported. Please pass a PIL.Image or file path string."
            )
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        img_embeds = self.blip_model.visual_encoder(img_tensor)
        img_atts = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(self.device)
        text_output = self.blip_model.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=img_embeds,
            encoder_attention_mask=img_atts,
            return_dict=True,
        )
        txt_features = text_output.last_hidden_state[:, 0, :].float()
        scores = self.mlp(txt_features)
        normalized_scores = (scores - self.mean) / self.std

        return normalized_scores.detach().cpu().numpy().item()

    def rank_inference(self, prompt, images_list):
        text_input = self.blip_model.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)
        txt_set = []
        for img in images_list:
            if isinstance(img, Image.Image):
                pil_img = img
            elif isinstance(img, str):
                if os.path.isfile(img):
                    pil_img = Image.open(img)
            else:
                raise TypeError(
                    "This image parameter type is not supported. Please pass a PIL.Image or file path string."
                )
            img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            img_embeds = self.blip_model.visual_encoder(img_tensor)
            img_atts = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(
                self.device
            )
            text_output = self.blip_model.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=img_embeds,
                encoder_attention_mask=img_atts,
                return_dict=True,
            )
            txt_set.append(text_output.last_hidden_state[:, 0, :])

        txt_features = torch.cat(txt_set, 0).float()
        scores = self.mlp(txt_features)
        normalized_scores = (scores - self.mean) / self.std
        normalized_scores = torch.squeeze(normalized_scores)
        _, ranks = torch.sort(normalized_scores, dim=0, descending=True)
        _, indices = torch.sort(ranks, dim=0)
        indices = indices + 1

        return (
            indices.detach().cpu().numpy().tolist(),
            normalized_scores.detach().cpu().numpy().tolist(),
        )
