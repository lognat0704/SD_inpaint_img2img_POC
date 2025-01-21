
import spaces
import torch
import random
from PIL import Image
import gradio as gr
from gradio_imageslider import ImageSlider
from masked_stable_diffusion_xl_img2img import MaskedStableDiffusionXLImg2ImgPipeline


pipeline = MaskedStableDiffusionXLImg2ImgPipeline.from_pretrained("frankjoshua/juggernautXL_v8Rundiffusion", dtype=torch.float16)

pipeline.to('cuda')
pipeline.enable_xformers_memory_efficient_attention()

MODELS = {
    "juggernautXL_v8Rundiffusion": "frankjoshua/juggernautXL_v8Rundiffusion"
}


@spaces.GPU(duration=24)
def fill_image(prompt, image, model_selection, mask_color):


    source = image["background"]
    source_rgb = source.convert("RGB")
    source_rgb.save("source.png", "PNG")

    mask = image["layers"][0]  
    alpha_channel = mask.split()[3]
    binary_mask = alpha_channel.point(lambda p: p > 0 and 255)
    cnet_image = source.copy()
    cnet_image.paste(0, (0, 0), binary_mask)

    # Create a new image with the selected color
    color_image = Image.new("RGB", source.size, mask_color)
    colored_mask = Image.composite(color_image, source.convert("RGB"), binary_mask)
    colored_mask.save("colored_mask.png")
    colored_mask = Image.open("colored_mask.png")

    #cnet_image = colored_mask.copy()
    seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    source = Image.open("source.png")

    
    result = pipeline(prompt=prompt, blur=25, image=colored_mask, original_image=source, strength=0.95,
                          generator=generator, num_inference_steps=30, num_images_per_prompt=1)
    im = result.images[0]
    
    yield source, im


def clear_result():
    return gr.update(value=None)


title = """<h1 align="center"> Mask and Inpaint & img2img</h1>
<div align="center">Draw the mask over part of the subject you want to change and write what you want to inpaint it with, your color prompt should eaxclaty the dolor picker you choose.</div>
"""

with gr.Blocks() as demo:
    gr.HTML(title)
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                info="Describe what to inpaint the mask with",
                lines=3,
            )
        with gr.Column():
            model_selection = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="juggernautXL_v8", 
                label="Model",
            )
            
            color_picker = gr.ColorPicker(label="color")

            with gr.Row():
                with gr.Column():
                    run_button = gr.Button("Generate")

    with gr.Row():
        input_image = gr.ImageMask(
            type="pil", label="Input Image", crop_size=(1024, 1024), layers=False
        )

        result = ImageSlider(
            interactive=False,
            label="Generated Image",
        )

    use_as_input_button = gr.Button("Use as Input Image", visible=False)

    def use_output_as_input(output_image):
        return gr.update(value=output_image[1])

    use_as_input_button.click(
        fn=use_output_as_input, inputs=[result], outputs=[input_image]
    )

    run_button.click(
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(
        fn=lambda: gr.update(visible=False),
        inputs=None,
        outputs=use_as_input_button,
    ).then(
        fn=fill_image,
        inputs=[prompt, input_image, model_selection, color_picker],
        outputs=result,
    ).then(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=use_as_input_button,
    )

    prompt.submit(
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(
        fn=lambda: gr.update(visible=False),
        inputs=None,
        outputs=use_as_input_button,
    ).then(
        fn=fill_image,
        inputs=[prompt, input_image, model_selection, color_picker],
        outputs=result,
    ).then(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=use_as_input_button,
    )


demo.queue(max_size=12).launch(share=True)
