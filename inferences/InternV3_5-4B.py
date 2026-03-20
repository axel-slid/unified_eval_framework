# %%

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = "OpenGVLab/InternVL3_5-4B-HF"
IMAGE_PATH = "tests/images/001.jpg"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    use_fast=False,
)

model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map=None,          # disable auto device mapping
    low_cpu_mem_usage=False,  # prevents meta tensor usage
)

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

model = model.eval().to(device)

image = Image.open(IMAGE_PATH).convert("RGB")

response = model.chat(
    tokenizer=tokenizer,
    image=image,
    question="<image>\nDescribe this image in detail.",
    generation_config={
        "max_new_tokens": 128,
        "do_sample": False,
    },
)

print(response)
# %%
