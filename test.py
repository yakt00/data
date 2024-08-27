from transformers import AutoTokenizer
from PIL import Image
from vllm import LLM, SamplingParams

MODEL_NAME = "openbmb/MiniCPM-V-2_6"
# Also available for previous models
# MODEL_NAME = "openbmb/MiniCPM-Llama3-V-2_5"
# MODEL_NAME = "HwwwH/MiniCPM-V-2"

# source_image = Image.open("../CIRR/examples/dev-150-3-img1.png").convert("RGB")
# img1 = Image.open("../CIRR/examples/dev-63-0-img1.png").convert("RGB")
# img2 = Image.open("../CIRR/examples/dev-150-2-img1.png").convert("RGB")
# img3 = Image.open("../CIRR/examples/dev-150-0-img1.png").convert("RGB")
# img4 = Image.open("../CIRR/examples/dev-430-3-img0.png").convert("RGB")
# img5 = '../DeepSeek-VL/examples/dev-244-0-img0.png'
# img6 = '../DeepSeek-VL/examples/dev-940-3-img0.png'
# img7 = '../DeepSeek-VL/examples/dev-138-3-img1.png'
# img8 = '../DeepSeek-VL/examples/dev-629-0-img0.png'
image = Image.open("../DeepSeek-VL/imgs/dev-150-3-img1.png").convert("RGB")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    gpu_memory_utilization=1,
    max_model_len=2048
)

messages = [{
    "role":
    "user",
    "content":
    # Number of images
    "(<image>./</image>)" + \
    "\nWhat is the content of this image?" 
}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Single Inference
inputs = {
    "prompt": prompt,
    "multi_modal_data": {
        "image": image
        # Multi images, the number of images should be equal to that of `(<image>./</image>)`
        # "image": [image, image] 
    },
}
# Batch Inference
# inputs = [{
#     "prompt": prompt,
#     "multi_modal_data": {
#         "image": image
#     },
# } for _ in 2]


# 2.6
stop_tokens = ['<|im_end|>', '<|endoftext|>']
stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
# 2.0
# stop_token_ids = [tokenizer.eos_id]
# 2.5
# stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]

sampling_params = SamplingParams(
    stop_token_ids=stop_token_ids, 
    use_beam_search=True,
    temperature=0, 
    best_of=3,
    max_tokens=1024
)

outputs = llm.generate(inputs, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
