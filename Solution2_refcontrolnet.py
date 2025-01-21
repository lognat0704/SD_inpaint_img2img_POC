from diffusers import ControlNetModel, AutoencoderKL
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch

import cv2
from PIL import Image

from stable_diffusion_xl_controlnet_reference import StableDiffusionXLControlNetReferencePipeline

# download an image
#canny_image = load_image(
#    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl_reference_input_cat.jpg"
#)

canny_image = load_image("shoe_mb.png")

#ref_image = load_image(
#    "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
#)

ref_image =load_image("shoe.jpg")

# initialize the models and pipeline
controlnet_conditioning_scale = 0.5  # recommended for good generalization
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetReferencePipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16
).to("cuda:0")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# get canny image
#image = np.array(canny_image)
#image = cv2.Canny(image, 100, 200)
#image = image[:, :, None]
#image = np.concatenate([image, image, image], axis=2)
#canny_image = Image.fromarray(image)
canny_image = ref_image
#canny_image.save("canny_image.png")
# generate image
image = pipe(
    prompt="A Nike shoe with blue rubber showbase.",
    num_inference_steps=20,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    image=canny_image,
    ref_image=ref_image,
    reference_attn=True,
    reference_adain=True,
    style_fidelity=1.0,
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]

image.save(f"result000.png")