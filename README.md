## Project Overview
This project explores different approaches to simulate main features from [Wand](https://www.instagram.com/wand.official/reel/DEdjsc4SPFF)



## Experimentation Process

#### Input:
Image A: original image

Image B: applied color masked on original image

#### Tryout 1: in-painting fill
The next approach involved in-painting using ControlNet's inpaint fill with the original resource parameters settings. Unsatisfactory results due to the difficulty of applying the color from the color mask to the masked area.

#### Tryout 2: Lineart + Reference Control
Initially, we experimented with Lineart + Reference Control. However, this method presented a problem: it was hard to keep unmasked areas the same as the original images.


#### Tryout 3 : Masked Im2Im Stable Diffusion
Finally, we adopted the Masked Im2Im Stable Diffusion Pipeline, which proved to be the most effective solution. We selected the Masked Im2Im Stable Diffusion Pipeline for this project. This pipeline is part of the Hugging Face Diffusers library and the code can be found in their [develop community](https://github.com/huggingface/diffusers/tree/main/examples/community#masked-im2im-stable-diffusion-pipeline)
