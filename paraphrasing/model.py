from typing import List
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


MAX_MAX_NEW_TOKENS = 200
TEMPERATURE = 0.9
TOP_P = 0.90
TOP_K = 50


def load_model(device, model_id="meta-llama/Llama-2-13b-chat-hf"):

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", cache_dir=".")
    tokenizer.pad_token = "<s>"
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        torch_dtype=torch.bfloat16,
        cache_dir="."
    ).to(device)

    return model, tokenizer


INSTRUCTION = "Paraphrase the following paragraphs, try to be very creative and make it look as different as possible without changing any meaning or losing any information. Don't be afraid to change the order of words or sentences. Don't add any new information that is not already in the text."


def get_prompt(paragraph: str, index=0) -> str:
    return f"<s>[INST] {INSTRUCTION}\n\n{paragraph.strip()} [/INST]"


@torch.inference_mode()
def paraphrase(model, tokenizer, paragraphs: List[str], device) -> str:
    prompts = [get_prompt(paragraph, 0) for i, paragraph in enumerate(paragraphs)]
    inputs = tokenizer(prompts, return_tensors='pt', add_special_tokens=False, padding=True).to(device)

    inputs.position_ids = torch.arange(inputs.input_ids.size(1), device=device)
    inputs.position_ids = inputs.position_ids.repeat(inputs.input_ids.size(0), 1)
    inputs.position_ids = (inputs.position_ids - (inputs.attention_mask == 0).sum(1, keepdim=True)).clamp(min=0) 

    generate_kwargs = dict(
        inputs,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_MAX_NEW_TOKENS,
        top_p=TOP_P,
        repetition_penalty=1.2,
        do_sample=True,
        use_cache=True
    )

    with torch.autocast(enabled=device != "cpu", device_type="cuda", dtype=torch.bfloat16):
        outputs = model.generate(**generate_kwargs)
    outputs = outputs.cpu()

    outputs = [
        tokenizer.decode(
            outputs[i, inputs.input_ids.size(1):],
            skip_special_tokens=True
        ).strip()
        for i in range(len(paragraphs))
    ]
    return outputs
