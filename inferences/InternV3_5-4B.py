# %%

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL_PATH = "OpenGVLab/InternVL3_5-4B-HF"
IMAGE_PATH = "inferences/images/001.jpg"

processor = AutoProcessor.from_pretrained(MODEL_PATH)

model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map=None,
    low_cpu_mem_usage=False,
)

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

model = model.eval().to(device)

image = Image.open(IMAGE_PATH).convert("RGB")

inputs = processor(
    text="<image>\nDescribe this image in detail.",
    images=image,
    return_tensors="pt",
).to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
# %%
