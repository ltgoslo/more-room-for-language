# coding=utf-8

import argparse
import torch
import torch.nn.functional as F
import gzip
import pickle
import tqdm
import json
from collections import Counter
from transformers import AutoTokenizer, AutoModelForMaskedLM
import wandb
import os
from tqdm import tqdm

from lm_score import rank


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_path", default="data", type=str, help="Path to BLiMP data directory.")
    parser.add_argument("--model_name", default="davda54/wiki-no-retrieval-base", type=str, help="The HuggingFace model name or local path to a HuggingFace implementation of the model.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size to use during inference.")

    args = parser.parse_args()

    return args


@torch.no_grad()
def evaluate(model, tokenizer, device, args):
    temperatures = torch.arange(0.0, 3.05, 0.05, device=device).clamp(min=1e-6)

    field_count = {"correct": [Counter() for _ in range(temperatures.size(0))], "total": [Counter() for _ in range(temperatures.size(0))]}
    uid_count = {"correct": [Counter() for _ in range(temperatures.size(0))], "total": [Counter() for _ in range(temperatures.size(0))]}
    linguistics_term_count = {"correct": [Counter() for _ in range(temperatures.size(0))], "total": [Counter() for _ in range(temperatures.size(0))]}


    # iterate through all .jsonl files in data directory
    for filename in os.listdir(args.input_path):
        if not filename.endswith(".jsonl"):
            continue

        # open file
        with open(os.path.join(args.input_path, filename), "r") as file:
            # iterate through each line in file
            for line in tqdm(file):
                # parse line
                line = json.loads(line.strip())

                # add to pairs
                pair = {
                    "good": line["sentence_good"],
                    "bad": line["sentence_bad"],
                    "field": line["field"],
                    "UID": line["UID"],
                    "linguistics_term": line["linguistics_term"]
                }
                if pair["field"] == "syntax_semantics":
                    pair["field"] = "syntax/semantics"

                # rank
                _, finegrained_ranking = rank([pair["good"], pair["bad"]], model, tokenizer, device, args.batch_size, temperatures=temperatures)

                for i, ranking in enumerate(finegrained_ranking): 
                    if ranking[0] == 0:
                        field_count["correct"][i][pair["field"]] += 1
                        uid_count["correct"][i][pair["UID"]] += 1
                        linguistics_term_count["correct"][i][pair["linguistics_term"]] += 1
                    field_count["total"][i][pair["field"]] += 1
                    uid_count["total"][i][pair["UID"]] += 1
                    linguistics_term_count["total"][i][pair["linguistics_term"]] += 1

    # compute accuracy

    field_accuracy = [{key: field_count["correct"][i][key] / field_count["total"][i][key] * 100.0 for key in field_count["correct"][i].keys()} for i in range(len(finegrained_ranking))]
    uid_accuracy = [{key: uid_count["correct"][i][key] / uid_count["total"][i][key] * 100.0 for key in uid_count["correct"][i].keys()} for i in range(len(finegrained_ranking))]
    linguistics_term_accuracy = [{key: linguistics_term_count["correct"][i][key] / linguistics_term_count["total"][i][key] * 100.0 for key in linguistics_term_count["correct"][i].keys()} for i in range(len(finegrained_ranking))]

    average_accuracies = [sum(uid_accuracy[i].values()) / len(uid_accuracy[i].values()) for i in range(len(finegrained_ranking))]

    for temperature, acc in zip(temperatures.tolist(), average_accuracies):
        print(f"{temperature}\t{acc:.2f}")
    print()

    average_accuracies = torch.tensor(average_accuracies)
    max_temp = torch.argmax(average_accuracies)
    print(f"BEST TEMPERATURE: {max_temp * 0.05}")
    print()

    # print
    print("### FIELD ACCURACY")
    for key in field_accuracy[max_temp].keys():
        print(f"{key}: {field_accuracy[max_temp][key]:.2f}")
    print()
    
    print("### LINGUISTIC TERM ACCURACY")
    for key in linguistics_term_accuracy[max_temp].keys():
        print(f"{key}: {linguistics_term_accuracy[max_temp][key]:.2f}")
    print()

    print("### UID ACCURACY")
    for key in uid_accuracy[max_temp].keys():
        print(f"{key}: {uid_accuracy[max_temp][key]:.2f}")
    print()

    print("### AVERAGE ACCURACY")
    print(f"{average_accuracies[max_temp]:.2f}")
    print()

    # save report
    with open(f"report_{args.model_name.split('/')[-1]}.txt", "w") as file:
        file.write("### FIELD ACCURACY\n")
        for key in field_accuracy[max_temp].keys():
            file.write(f"{key}: {field_accuracy[max_temp][key]:.2f}\n")
        file.write("\n")
        
        file.write("### LINGUISTIC TERM ACCURACY\n")
        for key in linguistics_term_accuracy[max_temp].keys():
            file.write(f"{key}: {linguistics_term_accuracy[max_temp][key]:.2f}\n")
        file.write("\n")

        file.write("### UID ACCURACY\n")
        for key in uid_accuracy[max_temp].keys():
            file.write(f"{key}: {uid_accuracy[max_temp][key]:.2f}\n")
        file.write("\n")

        file.write("### AVERAGE ACCURACY\n")
        file.write(f"{average_accuracies[max_temp]:.2f}\n")
        file.write("\n")


if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name, trust_remote_code=True).to(device)
    model.eval()

    evaluate(model, tokenizer, device, args)
