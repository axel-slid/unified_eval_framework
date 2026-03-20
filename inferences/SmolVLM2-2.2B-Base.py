# %%

# %%
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_PATH = "HuggingFaceTB/SmolVLM2-2.2B-Base"  # or local path to your downloaded model
IMAGE_PATH = "tests/images/001.jpg"

processor = AutoProcessor.from_pretrained(MODEL_PATH)

model = AutoModelForVision2Seq.from_pretrained(
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

# SmolVLM uses a chat template with messages format
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": IMAGE_PATH},
            {"type": "text",  "text": "Describe this image in detail."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
).to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

# Decode only the newly generated tokens (skip the input prompt)
generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
response = processor.decode(generated_ids[0], skip_special_tokens=True)
print(response)
# %%