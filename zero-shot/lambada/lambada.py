import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from tqdm import tqdm
from ftfy import fix_text
import argparse
import json

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # Required Parameters
    parser.add_arguments("--data", default="lambada.jsonl", type=str, help="Path to file containing the lambada dataset, we expect it to be in a JSONL format.")
    parser.add_arguments("--model_name_or_path", default="bert-base-cased", type=str, help="Either the HuggingFace name or local path to the model, we expect it to be a HuggingFace implementation.")
    
    # Optinal Parameters
    parser.add_arguments("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Outputs the prompt, answer and prediction of the model. Stops after num_prompts prompts.")
    parser.add_arguments("--num_prompts", default=10, type=int, help="Number of verbose prompts to output. Only used when verbose is True.")

    args = parser.parse_args()

    return args

args = parse_arguments()

with open(args.data, "r") as f:
    new_texts = [json.loads(line) for line in f if len(line.strip()) > 0]

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)

model.eval()

verbose = args.verbose
num_prompts = args.num_prompts

correct_answers = 0
total_answers = 0
perplexity = 0.0

progress_bar = tqdm(new_texts)

for i, text in enumerate(progress_bar):
    answer = text["answer"]
    prompt = text["prompt"]

    prompt, ending = prompt.split("{answer}")

    if verbose:
        print(f"Prompt: {prompt}")
        print(f"Answer: {answer}")
        print(f"Ending: {ending}")

    inputs = tokenizer(prompt.strip(), add_special_tokens=False)["input_ids"]
    ending = tokenizer(f'#{ending.strip()}', add_special_tokens=False)["input_ids"][1:]
    gold_output = tokenizer(answer, return_tensors="pt", add_special_tokens=False)["input_ids"]

    inputs = [tokenizer.cls_token_id] + inputs + [tokenizer.mask_token_id] * gold_output.size(1) + ending + [tokenizer.sep_token_id]
    inputs = torch.tensor(inputs).unsqueeze(0)

    if verbose:
        print(f"Prompt: {inputs}")
        print(f"Ending: {ending}")
        print(f"Answer: {gold_output}")

    with torch.no_grad():
        logits = model(inputs).logits[0, -(gold_output.size(1) + len(ending) + 1):-(len(ending) + 1)]
        loss = F.cross_entropy(logits, gold_output[0])

    prediction = tokenizer.decode(logits.argmax(-1))
    perplexity += loss

    if verbose:
        print(f"Prediction: {prediction}")
        print()
    
    if prediction.strip() == answer.strip():
        correct_answers += 1
    elif verbose:
        print(f"Wrong answer: {prediction} != {answer}")

    total_answers += 1

    accuracy = correct_answers/total_answers * 100.0
    avg_perplexity = torch.exp(perplexity/total_answers)

    progress_bar.set_description(f"Accuracy: {accuracy:.2f}%, Perplexity: {avg_perplexity:.2f}")

    if verbose and i == num_prompts:
        break

print(f"Accuracy: {correct_answers/total_answers * 100.0}")
print(f"Perplexity: {torch.exp(perplexity/total_answers)}")

