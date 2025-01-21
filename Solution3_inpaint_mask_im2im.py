from PIL import Image
import torch
from masked_stable_diffusion_xl_img2img import MaskedStableDiffusionXLImg2ImgPipeline
img = Image.open("source.png")
# read image with mask painted over
img_paint = Image.open("colored_mask.png")



# Convert the image to RGBA mode (which includes an alpha channel)
#img_paint_rgba = img_paint.convert("RGBA")

# Create a new alpha channel (fully opaque)
#alpha = Image.new('L', img_paint.size, 255)

#img_paint.putalpha(alpha)


pipeline = MaskedStableDiffusionXLImg2ImgPipeline.from_pretrained("frankjoshua/juggernautXL_v8Rundiffusion", dtype=torch.float16)

pipeline.to('cuda')
pipeline.enable_xformers_memory_efficient_attention()

prompt = "shoe rubber the is blue"
seed = 8348273636437
for i in range(3):
    generator = torch.Generator(device="cuda").manual_seed(seed + i)
    print(seed + i)
    #prompt=prompt, blur=48, image=img_paint, original_image=img, strength=0.9,
    #generator=generator, num_inference_steps=60, num_images_per_prompt=1
    result = pipeline(prompt=prompt, blur=25, image=img_paint, original_image=img, strength=0.95,
                          generator=generator, num_inference_steps=60, num_images_per_prompt=1)
    im = result.images[0]
    im.save(f"result{i}.png")