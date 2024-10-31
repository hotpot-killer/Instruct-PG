import os
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm.auto import tqdm
from typing import List, Optional, Union
import inspect
import warnings
import torch

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
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

from packaging import version
import random
import ImageFlow
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from datasets import load_dataset
logger = logging.get_logger(__name__)

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
        # load llm
        self.llama_model = LlamaForCausalLM.from_pretrained(llm_path)
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llm_path)

    def train(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.Tensor, PIL.Image.Image],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        generator: Optional[torch.Generator] = None,
        embedding_learning_rate: float = 0.0001,
        diffusion_model_learning_rate: float = 2e-5,
        text_embedding_optimization_steps: int = 300,
        model_fine_tuning_optimization_steps: int = 800,
        **kwargs,
    ):
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="fp16",
        )

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

        if accelerator.is_main_process:
            accelerator.init_trackers(
                "InstructPG",
                config={
                    "embedding_learning_rate": embedding_learning_rate,
                    "text_embedding_optimization_steps": text_embedding_optimization_steps,
                },
            )
        # get better prompt from llm
        sys_prompt = """You are part of a team of bots that creates images. You work with an assistant bot that will draw anything you say in square brackets.

        For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" will trigger your partner bot to output an image of a forest morning, as described. You will be prompted by people looking to create detailed, amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.

        There are a few rules to follow:

        Single Image Description: You will only ever output a single image description per user request.
        Modifications: When modifications are requested, you should not simply make the description longer. You should refactor the entire description to integrate the suggestions.
        New Images: Other times the user will not want modifications, but instead want a new image. In this case, you should ignore your previous conversation with the user.
        Word Count: Image descriptions must have the same number of words as examples below. Extra words will be ignored.
        Example Prompts:

        "A beautiful morning in the woods with the sun peaking through the trees."
        "A bustling city street at night with bright neon lights and people walking."
        "A serene beach at sunset with waves gently lapping against the shore."

        """
        role = "You are an image editing robot, and you provide merge editing instruction to improve the functionality of the input provided. The editing commands should be as follows:"
        prompt = "".join(sys_prompt, prompt)

        inputs = self.llama_tokenizer(prompt, return_tensors="pt")
        # Generate
        generate_ids = self.llama_model.generate(inputs.input_ids, max_length=50)
        better_prompt = self.llama_tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

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
        text_embeddings = text_embeddings.detach()
        text_embeddings.requires_grad_()
        text_embeddings_orig = text_embeddings.clone()

        # Initialize the optimizer
        optimizer = torch.optim.Adam(
            [text_embeddings],  # only optimize the embeddings
            lr=embedding_learning_rate,
        )

        if isinstance(image, PIL.Image.Image):
            image = preprocess(image)

        latents_dtype = text_embeddings.dtype
        image = image.to(device=self.device, dtype=latents_dtype)
        init_latent_image_dist = self.vae.encode(image).latent_dist
        image_latents = init_latent_image_dist.sample(generator=generator)
        image_latents = 0.18215 * image_latents

        progress_bar = tqdm(
            range(text_embedding_optimization_steps),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")

        global_step = 0

        logger.info(
            "First optimizing the text embedding to better reconstruct the init image"
        )
        for _ in range(text_embedding_optimization_steps):
            with accelerator.accumulate(text_embeddings):
                # Sample noise that we'll add to the latents
                noise = torch.randn(image_latents.shape).to(image_latents.device)
                timesteps = torch.randint(800, (1,), device=image_latents.device)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = self.scheduler.add_noise(
                    image_latents, noise, timesteps
                )

                # Predict the noise residual
                noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample

                loss = (
                    F.l1_loss(noise_pred, noise, reduction="none")
                    .mean([1, 2, 3])
                    .mean()
                )
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {
                "loss": loss.detach().item()
            }  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        accelerator.wait_for_everyone()

        text_embeddings.requires_grad_(False)

        # Now we fine tune the unet to better reconstruct the image
        self.unet.requires_grad_(True)
        self.unet.train()
        optimizer = torch.optim.Adam(
            self.unet.parameters(),  # only optimize unet
            lr=diffusion_model_learning_rate,
        )
        progress_bar = tqdm(
            range(model_fine_tuning_optimization_steps),
            disable=not accelerator.is_local_main_process,
        )

        logger.info(
            "Next fine tuning the entire model to better reconstruct the init image"
        )
        for _ in range(model_fine_tuning_optimization_steps):
            with accelerator.accumulate(self.unet.parameters()):
                # Sample noise that we'll add to the latents
                noise = torch.randn(image_latents.shape).to(image_latents.device)
                timesteps = torch.randint(800, (1,), device=image_latents.device)
                noisy_latents = self.scheduler.add_noise(
                    image_latents, noise, timesteps
                )
                noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample

                loss = (
                    F.mse_loss(noise_pred, noise, reduction="none")
                    .mean([1, 2, 3])
                    .mean()
                )
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {
                "loss": loss.detach().item()
            } 
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        accelerator.wait_for_everyone()
        self.text_embeddings_orig = text_embeddings_orig
        self.text_embeddings = text_embeddings

        self.preference_model = ImageFlow.load(
            "/data/image_editing/Instruct-PG/checkpoint/imageflow_state_dict.pt",
            device=accelerator.device,
        )

        # Freeze vae and text_encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.preference_model.requires_grad_(False)

        progress_bar = tqdm(
            range(global_step, 400),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")

        def tokenize_captions(examples, is_train=True):
            captions = []
            for caption in examples["prompt"]:
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    captions.append(random.choice(caption) if is_train else caption[0])
                else:
                    raise ValueError(
                        f"Caption column `{"prompt"}` should contain either strings or lists of strings."
                    )
            inputs = self.tokenizer(
                captions,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return inputs.input_ids

        def preprocess_train(examples):
            examples["input_ids"] = tokenize_captions(examples)
            examples["rm_input_ids"] = self.preference_model.blip_model.tokenizer(
                examples["prompt"],
                padding="max_length",
                truncation=True,
                max_length=35,
                return_tensors="pt",
            ).input_ids
            examples["rm_attention_mask"] = self.preference_model.blip_model.tokenizer(
                examples["prompt"],
                padding="max_length",
                truncation=True,
                max_length=35,
                return_tensors="pt",
            ).attention_mask
            return examples

        def collate_fn(examples):
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

        
        dataset = load_dataset("path of preference dataset")
        self.train_dataset = dataset["train"].with_transform(preprocess_train)
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=1,
            num_workers=1,
        )
        for epoch in range(0, 10):
            self.unet.train()
            train_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                # Skip steps until we reach the resumed step
                if epoch == 0:
                    if step % 4 == 0:
                        progress_bar.update(1)
                    continue

                with accelerator.accumulate(self.unet):
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
                    latents = torch.randn(
                        (2, 4, 64, 64),
                        device=accelerator.device,
                    )

                    self.scheduler.set_timesteps(40, device=accelerator.device)
                    timesteps = self.scheduler.timesteps

                    mid_timestep = torch.randint(45, 50)

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
                    ).pred_original_sample.to(self.weight_dtype)

                    pred_original_sample = (
                        1 / self.vae.config.scaling_factor * pred_original_sample
                    )
                    image = self.vae.decode(
                        pred_original_sample.to(self.weight_dtype)
                    ).sample
                    image = (image / 2 + 0.5).clamp(0, 1)

                    rm_preprocess = _transform()
                    image = rm_preprocess(image).to(accelerator.device)

                    rewards = self.preference_model.compute_scores(
                        batch["rm_input_ids"], batch["rm_attention_mask"], image
                    )
                    loss = F.relu(-rewards + 2)
                    loss = loss.mean() * 1e-3

                    avg_loss = accelerator.gather(loss.repeat(1)).mean()
                    train_loss += avg_loss.item() / 2

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
                    optimizer.step()
                    self.lr_scheduler.step()
                    optimizer.zero_grad()

                
                if accelerator.sync_gradients:
                    self.unet.step(self.unet.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    if global_step % 100 == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join("./", f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)

                if global_step >= 100:
                    break

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
        if self.text_embeddings is None:
            raise ValueError(
                "Please run the pipe.train() before trying to generate an image."
            )
        if self.text_embeddings_orig is None:
            raise ValueError(
                "Please run the pipe.train() before trying to generate an image."
            )

        text_embeddings = (
            alpha * self.text_embeddings_orig + (1 - alpha) * self.text_embeddings
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

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents_shape = (1, self.unet.config.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype
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
                latent_model_input, t, encoder_hidden_states=text_embeddings
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
