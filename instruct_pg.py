import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm.auto import tqdm
from typing import List, Optional, Union
import inspect
import warnings
import torch.nn as nn

from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
)
from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import StableDiffusionMixin
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import logging

from diffusers.training_utils import EMAModel
from packaging import version
import random
import ImageFlow
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import torchvision
from datasets import load_dataset

logger = logging.get_logger(__name__)
from diffusers.optimization import get_scheduler

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


def preprocess(image):
    w, h = image.size
    w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


# image encode
def _transform():
    return Compose(
        [
            Resize(224, interpolation=PIL_INTERPOLATION["bicubic"]),
            CenterCrop(224),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


from dataclasses import dataclass


@dataclass
class InstructPGConfig:
    embedding_learning_rate: float = 0.001
    diffusion_model_learning_rate: float = 2e-6
    text_embedding_optimization_steps: int = 500
    model_fine_tuning_optimization_steps: int = 1000
    batch_size: int = 1
    num_workers: int = 1
    gradient_accumulation_steps: int = 4


class PerceptualLoss(nn.Module):
    """感知损失"""

    def __init__(self):
        super().__init__()
        # 使用预训练的VGG16提取特征
        vgg = torchvision.models.vgg16(pretrained=True).features
        self.blocks = nn.ModuleList([
            vgg[:4],  # conv1_2
            vgg[4:9],  # conv2_2
            vgg[9:16],  # conv3_3
            vgg[16:23]  # conv4_3
        ])
        for bl in self.blocks:
            for p in bl.parameters():
                p.requires_grad = False

    def forward(self, x, y):
        # 归一化到VGG的输入范围
        x = (x + 1) * 0.5
        y = (y + 1) * 0.5

        # 计算不同层的特征差异
        loss = 0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss


class CombinedLoss(nn.Module):
    """组合损失"""

    def __init__(self, lambda_l1=1.0, lambda_perceptual=0.1):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.perceptual_loss = PerceptualLoss()

    def forward(self, pred, target, images=None):
        # L1 损失
        l1_loss = F.l1_loss(pred, target)

        # 如果提供了图像，计算感知损失
        perceptual_loss = 0
        if images is not None:
            perceptual_loss = self.perceptual_loss(pred, target)

        # 组合损失
        total_loss = self.lambda_l1 * l1_loss + \
                     self.lambda_perceptual * perceptual_loss

        return total_loss, {
            'l1_loss': l1_loss.item(),
            'perceptual_loss': perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else 0
        }


class InstructPGStableDiffusionPipeline(DiffusionPipeline, StableDiffusionMixin):
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            llm_path="meta-llama/Llama-2-7b-chat-hf",
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        dataset = load_dataset(
            "json", data_files="/data/wzh/dataset/ImageFlowData/hf_img_edit_preference.jsonl"
        )
        # DataLoaders creation:
        self.train_dataset = dataset["train"].with_transform(self.preprocess_train)
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=1,
            num_workers=1,
        )
        self.preference_model = None
        self.accelerator = Accelerator(
            gradient_accumulation_steps=4,
            mixed_precision="fp16"
        )
        self.reward_optimizer = torch.optim.Adam(
            self.unet.parameters(),
            lr=1e-05
        )

        self.lr_scheduler = get_scheduler(
            "constant",
            optimizer=self.reward_optimizer,
            num_warmup_steps=0 * 4,
            num_training_steps=100 * 4,
        )
        # self.llama_model = LlamaForCausalLM.from_pretrained(llm_path)
        # self.llama_tokenizer = LlamaTokenizer.from_pretrained(llm_path)

    def save_model(self, path):
        """保存模型状态"""
        torch.save({
            'unet_state_dict': self.unet.state_dict(),
            'text_embeddings_orig': self.text_embeddings_orig,
            'text_embeddings_optim': self.text_embeddings_optim,
            'optimizer_state_dict': self.reward_optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
        }, path)

    def load_model(self, path):
        """加载模型状态"""
        checkpoint = torch.load(path)
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        self.text_embeddings_orig = checkpoint['text_embeddings_orig']
        self.text_embeddings_optim = checkpoint['text_embeddings_optim']
        self.reward_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def tokenize_captions(self, examples, is_train=True):
        captions = []
        for caption in examples["prompt"]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{{'caption'}}` should contain either strings or lists of strings."
                )
        inputs = self.tokenizer(
            captions,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    def preprocess_train(self, examples):
        examples["input_ids"] = self.tokenize_captions(examples)
        examples["rm_input_ids"] = self.preference_model.blip.tokenizer(
            examples["prompt"],
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).input_ids
        examples["rm_attention_mask"] = self.preference_model.blip.tokenizer(
            examples["prompt"],
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).attention_mask
        return examples

    def collate_fn(self, examples):
        input_ids = torch.stack([example["input_ids"] for example in examples])
        rm_input_ids = torch.stack(
            [example["rm_input_ids"] for example in examples]
        )
        rm_attention_mask = torch.stack(
            [example["rm_attention_mask"] for example in examples]
        )
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        rm_input_ids = rm_input_ids.view(-1, rm_input_ids.shape[-1])
        rm_attention_mask = rm_attention_mask.view(-1, rm_attention_mask.shape[-1])
        return {
            "input_ids": input_ids,
            "rm_input_ids": rm_input_ids,
            "rm_attention_mask": rm_attention_mask,
        }

    def train(
            self,
            prompt: Union[str, List[str]],
            image: Union[torch.Tensor, PIL.Image.Image],
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            generator: Optional[torch.Generator] = None,
            embedding_learning_rate: float = 0.001,
            diffusion_model_learning_rate: float = 2e-6,
            text_embedding_optimization_steps: int = 500,
            model_fine_tuning_optimization_steps: int = 1000,
            **kwargs,
    ):
        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        # Freeze vae and unet
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                "InstructPG",
                config={
                    "embedding_learning_rate": embedding_learning_rate,
                    "text_embedding_optimization_steps": text_embedding_optimization_steps,
                },
            )
        # # get better prompt from llm
        # sys_prompt = """You are part of a team of bots that creates images. You work with an assistant bot that will draw anything you say in square brackets.

        # For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" will trigger your partner bot to output an image of a forest morning, as described. You will be prompted by people looking to create detailed, amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.

        # There are a few rules to follow:

        # Single Image Description: You will only ever output a single image description per user request.
        # Modifications: When modifications are requested, you should not simply make the description longer. You should refactor the entire description to integrate the suggestions.
        # New Images: Other times the user will not want modifications, but instead want a new image. In this case, you should ignore your previous conversation with the user.
        # Word Count: Image descriptions must have the same number of words as examples below. Extra words will be ignored.
        # Example Prompts:

        # "A beautiful morning in the woods with the sun peaking through the trees."
        # "A bustling city street at night with bright neon lights and people walking."
        # "A serene beach at sunset with waves gently lapping against the shore."

        # """
        # role = "You are an image editing robot, and you provide merge editing instruction to improve the functionality of the input provided. The editing commands should be as follows:"
        # prompt = "".join(sys_prompt, prompt)

        # inputs = self.llama_tokenizer(prompt, return_tensors="pt")
        # # Generate
        # generate_ids = self.llama_model.generate(inputs.input_ids, max_length=50)
        # better_prompt = self.llama_tokenizer.batch_decode(
        #     generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )[0]

        # get text embeddings for prompt
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = torch.nn.Parameter(
            self.text_encoder(text_input.input_ids.to(self.device))[0],
            requires_grad=True,
        )
        text_embeddings_optim = text_embeddings.detach()
        text_embeddings_optim.requires_grad_()
        text_embeddings_orig = text_embeddings_optim.clone()

        # Initialize the optimizer
        optimizer = torch.optim.Adam(
            [text_embeddings_optim],  # only optimize the embeddings
            lr=embedding_learning_rate,
        )

        if isinstance(image, PIL.Image.Image):
            image = preprocess(image)

        latents_dtype = text_embeddings_optim.dtype
        image = image.to(device=self.device, dtype=latents_dtype)
        init_latent_image_dist = self.vae.encode(image).latent_dist
        image_latents = init_latent_image_dist.sample(generator=generator)
        image_latents = 0.18215 * image_latents

        progress_bar = tqdm(
            range(text_embedding_optimization_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")

        global_step = 0

        logger.info(
            "First optimizing the text embedding to better reconstruct the init image"
        )
        # 1. Optimize the text embedding

        for _ in range(text_embedding_optimization_steps):
            with self.accelerator.accumulate(text_embeddings_optim):
                # Sample noise that we'll add to the latents
                noise = torch.randn(image_latents.shape).to(self.device)
                timesteps = torch.randint(1000, (1,), device=self.device)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = self.scheduler.add_noise(
                    image_latents, noise, timesteps
                )

                # Predict the noise residual
                noise_pred = self.unet(noisy_latents, timesteps, text_embeddings_optim).sample

                loss = (
                    F.mse_loss(noise_pred, noise, reduction="none")
                    .mean([1, 2, 3])
                    .mean()
                )
                self.accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {
                "loss": loss.detach().item()
            }
            progress_bar.set_postfix(**logs)
            self.accelerator.log(logs, step=global_step)

        self.accelerator.wait_for_everyone()

        text_embeddings_optim.requires_grad_(False)

        # 2. fine tune the unet to better reconstruct the image
        self.unet.requires_grad_(True)
        self.unet.train()
        optimizer = torch.optim.Adam(
            self.unet.parameters(),  # only optimize unet
            lr=diffusion_model_learning_rate,
        )
        progress_bar = tqdm(
            range(model_fine_tuning_optimization_steps),
            disable=not self.accelerator.is_local_main_process,
        )

        logger.info(
            "Next fine tuning the entire model to better reconstruct the init image"
        )
        for _ in range(model_fine_tuning_optimization_steps):
            with self.accelerator.accumulate(self.unet.parameters()):
                # Sample noise that we'll add to the latents
                noise = torch.randn(image_latents.shape).to(self.device)
                timesteps = torch.randint(1000, (1,), device=self.device)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = self.scheduler.add_noise(
                    image_latents, noise, timesteps
                )

                # Predict the noise residual
                noise_pred = self.unet(noisy_latents, timesteps, text_embeddings_optim).sample

                loss = (
                    F.mse_loss(noise_pred, noise, reduction="none")
                    .mean([1, 2, 3])
                    .mean()
                )
                self.accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {
                "loss": loss.detach().item()
            }
            progress_bar.set_postfix(**logs)
            self.accelerator.log(logs, step=global_step)

        self.accelerator.wait_for_everyone()
        self.text_embeddings_orig = text_embeddings_orig
        self.text_embeddings_optim = text_embeddings_optim

    def train_image_flow(self,
                         diffusion_model_learning_rate: float = 2e-6):
        # 添加全局步数计数器
        global_step = 0
        # Fine tuning the Unet using reward model
        self.preference_model = ImageFlow.load(
            "/data/wzh/image_editing/Instruct-PG/checkpoint/reward_diffusion_state_dict.pt",
            device=self.device,
        )
        # Freeze vae and text_encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.preference_model.requires_grad_(False)

        self.unet.train()

        optimizer = torch.optim.Adam(
            self.unet.parameters(),  # only optimize unet
            lr=diffusion_model_learning_rate,
        )

        progress_bar = tqdm(
            range(global_step, 100),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")
        logs = {
            "start to preference optimization"
        }
        for epoch in range(0, 100):
            train_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                # Skip steps until we reach the resumed step
                if epoch == 0:
                    if step % 4 == 0:
                        progress_bar.update(1)
                    continue

                with self.accelerator.accumulate(self.unet):
                    encoder_hidden_states = self.text_encoder(batch["input_ids"].to(self.device))[0]
                    latents = torch.randn(
                        (1, 4, 64, 64),
                        device=self.device,
                    )

                    self.scheduler.set_timesteps(50, device=self.device)
                    timesteps = self.scheduler.timesteps

                    mid_timestep = random.randint(45, 49)

                    for i, t in enumerate(timesteps[:mid_timestep]):
                        with torch.no_grad():
                            latent_model_input = latents
                            latent_model_input = self.scheduler.scale_model_input(
                                latent_model_input, t
                            )
                            noise_pred = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=encoder_hidden_states,
                            ).sample
                            latents = self.scheduler.step(
                                noise_pred, t, latents
                            ).prev_sample

                    latent_model_input = latents
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, timesteps[mid_timestep]
                    )
                    noise_pred = self.unet(
                        latent_model_input,
                        timesteps[mid_timestep],
                        encoder_hidden_states=encoder_hidden_states,
                    ).sample
                    pred_original_sample = self.scheduler.step(
                        noise_pred, timesteps[mid_timestep], latents
                    ).pred_original_sample.to(self.text_embeddings_optim.dtype)

                    pred_original_sample = (
                            1 / self.vae.config.scaling_factor * pred_original_sample
                    )
                    image = self.vae.decode(
                        pred_original_sample.to(self.text_embeddings_optim.dtype)
                    ).sample
                    image = (image / 2 + 0.5).clamp(0, 1)

                    rm_preprocess = _transform()
                    image = rm_preprocess(image).to(self.device)
                    rewards = self.preference_model.score_gard(
                        batch["rm_input_ids"].to(self.device), batch["rm_attention_mask"].to(self.device), image
                    )
                    loss = F.relu(-rewards + 2)
                    loss = loss.mean() * 1e-3

                    avg_loss = self.accelerator.gather(loss.repeat(1)).mean()
                    train_loss += avg_loss.item() / 2

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
                    optimizer.step()
                    self.lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                    "global_step": global_step,
                }
                progress_bar.set_postfix(**logs)

                if global_step >= 30:
                    break
        self.accelerator.end_training()

    @torch.no_grad()
    def __call__(
            self,
            alpha: float = 1.2,
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            num_inference_steps: Optional[int] = 50,
            generator: Optional[torch.Generator] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            guidance_scale: float = 7.5,
            eta: float = 0.0,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )
        if self.text_embeddings_optim is None:
            raise ValueError(
                "Please run the pipe.train() before trying to generate an image."
            )
        if self.text_embeddings_orig is None:
            raise ValueError(
                "Please run the pipe.train() before trying to generate an image."
            )

        self.text_embeddings = (
                alpha * self.text_embeddings_orig + (1 - alpha) * self.text_embeddings_optim
        )

        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            uncond_tokens = [""]
            max_length = self.tokenizer.model_max_length
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.view(1, seq_len, -1)

            self.text_embeddings = torch.cat([uncond_embeddings, self.text_embeddings])

        latents_shape = (1, self.unet.config.in_channels, height // 8, width // 8)
        latents_dtype = self.text_embeddings.dtype
        if self.device.type == "mps":
            # randn does not exist on mps
            latents = torch.randn(
                latents_shape, generator=generator, device="cpu", dtype=latents_dtype
            ).to(self.device)
        else:
            latents = torch.randn(
                latents_shape,
                generator=generator,
                device=self.device,
                dtype=latents_dtype,
            )

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        latents = latents * self.scheduler.init_noise_sigma
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=self.text_embeddings
            ).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)
        if not return_dict:
            return image

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)