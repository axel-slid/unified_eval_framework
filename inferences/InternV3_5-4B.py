# %%

import torch
from PIL import Image
from transformers import AutoProcessor, InternVLForConditionalGeneration

MODEL_PATH = "OpenGVLab/InternVL3_5-4B-HF"
IMAGE_PATH = "inferences/images/001.jpg"

processor = AutoProcessor.from_pretrained(MODEL_PATH)

model = InternVLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

model = model.eval()

image = Image.open(IMAGE_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]

text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

inputs = processor(
    text=text_prompt,
    images=image,
    return_tensors="pt",
).to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
# %%
