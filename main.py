import torch
import os
from PIL import Image
from diffusers import DDIMScheduler
from Instruct_pg import InstructPGStableDiffusionPipeline
from diffusers.utils import load_image

pipeline = InstructPGStableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    scheduler=DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    ),
).to("cuda")

generator = torch.Generator("cuda").manual_seed(0)
seed = 0
prompt = "Giclee Print - Seven Crows in the Marsh - 20 x 30"
url = "./data/prompt.png"
source_image = load_image(url)
width, height = source_image.size
min_dimension = min(width, height)

left = (width - min_dimension) / 2
top = (height - min_dimension) / 2
right = (width + min_dimension) / 2
bottom = (height + min_dimension) / 2

final_source = source_image.crop((left, top, right, bottom))
final_source = final_source.resize((512, 512), Image.LANCZOS)

train_pipe = pipeline.train(
    prompt,
    image=final_source,
    generator=generator,
    text_embedding_optimization_steps=10,
    model_fine_tuning_optimization_steps=10,
)

images = train_pipe(guidance_scale=7.5, num_inference_steps=50).images
os.makedirs("output", exist_ok=True)
image = images[0]
image.save("./output/image.png")