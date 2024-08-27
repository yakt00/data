import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

torch.manual_seed(0)

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

# source_image = Image.open("../CIRR/examples/dev-150-3-img1.png").convert("RGB")
# img1 = Image.open("../CIRR/examples/dev-63-0-img1.png").convert("RGB")
# img2 = Image.open("../CIRR/examples/dev-150-2-img1.png").convert("RGB")
# img3 = Image.open("../CIRR/examples/dev-150-0-img1.png").convert("RGB")
# img4 = Image.open("../CIRR/examples/dev-430-3-img0.png").convert("RGB")
# img5 = '../DeepSeek-VL/examples/dev-244-0-img0.png'
# img6 = '../DeepSeek-VL/examples/dev-940-3-img0.png'
# img7 = '../DeepSeek-VL/examples/dev-138-3-img1.png'
# img8 = '../DeepSeek-VL/examples/dev-629-0-img0.png'

image = Image.open('../DeepSeek-VL/imgs/dev-150-3-img1.png').convert('RGB')

# First round chat 
question = "Describe the image."
msgs = [{'role': 'user', 'content': [image, question]}]

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)

# # Second round chat 
# # pass history context of multi-turn conversation
# msgs.append({"role": "assistant", "content": [answer]})
# msgs.append({"role": "user", "content": ["Introduce something about Airbus A380."]})

# answer = model.chat(
#     image=None,
#     msgs=msgs,
#     tokenizer=tokenizer
# )
# print(answer)
