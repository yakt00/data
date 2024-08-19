import os
import copy
import json
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images


def clean_response(response):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response

def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response

model_path = "deepseek-ai/deepseek-vl-7b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

fig_path = 'concate_img'
source_image = 'imgs/dev-150-3-img1.png'
img1 = 'imgs/dev-63-0-img1.png'
img2 = 'imgs/dev-150-2-img1.png'
img3 = 'imgs/dev-150-0-img1.png'
img4 = 'imgs/dev-430-3-img0.png'
img5 = 'imgs/dev-244-0-img0.png'
img6 = 'imgs/dev-940-3-img0.png'
img7 = 'imgs/dev-138-3-img1.png'
img8 = 'imgs/dev-629-0-img0.png'

conversation = [
            {
                "role": "system",
                "content": "You are a intelligent assistant ranker that can rank the gallery images based on their relavence to the edited source image modified by the text instruction."

            },
            {
                "role": "user",
                "content": "I will provide you with 4 gallary images, each indicated by number identifier []. \nRank the gallary images based on their relevance to source image <image_placeholder> edited by instruction text 'be a same breed dog with his puppy running'."
                           "gallery image [1]: <image_placeholder>."
                           "gallery image [2]: <image_placeholder>."
                           "gallery image [3]: <image_placeholder>."
                           "gallery image [4]: <image_placeholder>."
                           "Rank the 4 gallery images above based on their relevance to the source image modified by instruction text. The gallery images should be listed in descending order using identifiers. The most relevant images should be listed first. The output format should be [] > [] > [] > [], e.g., [1] > [2] > [3] > [4]. Only response the ranking results, do not say any word or explain.",
                "images": [source_image, img1, img2, img3, img4,],
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
).to(vl_gpt.device)
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True,
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)

import pdb
pdb.set_trace()
print(f"{prepare_inputs['sft_format'][0]}", answer)

