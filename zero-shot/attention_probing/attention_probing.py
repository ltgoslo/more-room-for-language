from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

import dependency_decoding

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_arguments("--model_name_or_path", default="bert-base-cased", type=str, help="Either the HuggingFace model name or local path to a HuggingFace implementation of the model.")
    parser.add_arguments("--data", default="en_ewt-ud-dev.conllu", type=str, help="Path to the dataset file. We expect it to be in a conllu format.")
    parser.add_arguments("--num_attention_heads", default=12, type=int, help="The number of attention heads per layer in the model.")
    parser.add_arguments("--num_layers", default=12, type=int, help="The number of layers in the model.")    

    args = parser.parse_args()

    return args

args = parse_arguments()

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModel.from_pretrained(args.model_name_or_path, output_hidden_states=True, output_attentions=True, return_dict=True, trust_remote_code=True)

model.eval()

entries, current = [], []
for line in open(args.data):
    if line.startswith("#"):
        continue

    line = line.strip()

    if len(line) == 0:
        if len(current) == 0:
            continue
        entries.append(current)
        current = []
        continue

    res = line.split("\t")
    if not res[0].isdigit():
        continue
    current.append(res)

vocab_size = tokenizer.vocab_size
pad_index = tokenizer.pad_token_id
add_sep = True

forms = [[current[1] for current in entry] for entry in entries]

all_subwords, all_alignment = [], []
n_splits, n = 0, 0
for i_sentence, sentence in enumerate(forms):
    subwords, alignment = [tokenizer.cls_token_id], [0]
    for i, word in enumerate(sentence):
        space_before = (i == 0) or (not "SpaceAfter=No" in entries[i_sentence][i - 1][-1])

        # very very ugly hack ;(
        encoding = tokenizer(f"| {word}" if space_before else f"|{word}", add_special_tokens=False)
        subwords += encoding.input_ids[1:]
        alignment += (len(encoding.input_ids) - 1) * [i + 1]

        assert len(encoding.input_ids) > 1, f"{word} {encoding.input_ids}"

        if not word.isalpha():
            continue
        n_splits += len(encoding.input_ids) - 2
        n += 1

    if add_sep:
        subwords.append(tokenizer.sep_token_id)
        alignment.append(alignment[-1] + 1)

    all_subwords.append(subwords)
    all_alignment.append(alignment)

subwords = all_subwords
alignment = all_alignment
edges = [[int(current[6]) for current in entry] for entry in entries]

N_HEADS = args.num_attention_heads
N_LAYERS = args.num_layers

n_correct, n_total = {}, {}
for layer in range(N_LAYERS):
    for head in range(N_HEADS):
        n_correct[f"head_{layer}_{head}"] = 0
        n_total[f"head_{layer}_{head}"] = 0
    n_correct[f"layer_{layer}"] = 0
    n_total[f"layer_{layer}"] = 0
n_correct["average"] = 0
n_total["average"] = 0

for i in tqdm(range(len(forms))):
    alignment_matrix = F.one_hot(torch.tensor(alignment[i]), num_classes=len(forms[i]) + 2).float()

    with torch.no_grad():
        output = model(torch.tensor([subwords[i]]))
        attentions = output.attentions

        attentions_dict = {}
        for layer in range(N_LAYERS):
            for head in range(N_HEADS):
                attentions_dict[f"head_{layer}_{head}"] = attentions[layer][0, head, :, :]
            attentions_dict[f"layer_{layer}"] = attentions[layer][0, :, :, :].mean(dim=0)
        attentions_dict["average"] = torch.cat(attentions).flatten(0, 1).mean(0)

        for key, attention in attentions_dict.items():
            # align subwords to words by averaging
            attention = alignment_matrix.T @ attention @ alignment_matrix  # shape (num_words, num_words)
            attention = attention / alignment_matrix.sum(0)  # shape (num_words, num_words)

            attention = attention * attention.T  # shape (num_subwords, num_subwords)

            attention = attention[1:-1, 1:-1]

            attentions_dict[key] = attention
        
        predicted_edges = {
            key: dependency_decoding.chu_liu_edmonds(attention.numpy().astype(float))[0]
            for key, attention in attentions_dict.items()
        }

        gold_edges = {
            tuple(sorted([a, b - 1]))
            for a, b in enumerate(edges[i])
            if b > 0
        }

        predicted_edges = {
            key: {
                tuple(sorted([a, b]))
                for a, b in enumerate(parents)
                if b > 0
            }
            for key, parents in predicted_edges.items()
        }

        for key in predicted_edges:
            n_correct[key] += len(gold_edges & predicted_edges[key])
            n_total[key] += len(gold_edges)


accuracies = {
    key: n_correct[key] / n_total[key]
    for key in n_correct
}

results = accuracies

# Preparing individual data sets for subplots
head_data = np.array([[results[f"head_{i}_{j}"] * 100 for j in range(N_HEADS)] for i in range(N_LAYERS)])
layer_data = np.array([[results[f"layer_{i}"] * 100] for i in range(N_LAYERS)])
average_data = np.full((1, 1), results["average"] * 100)

print(f"average UUAS: {head_data.mean():.1f}")
print(f"median UUAS: {np.median(head_data):.1f}")
print(f"max UUAS: {np.max(head_data):.1f}")

# Creating the plot with subplots
fig, axs = plt.subplots(N_LAYERS, 3, figsize=((N_HEADS + 3) * 0.6, 7), gridspec_kw={'width_ratios': [N_HEADS, 1, 1]})  # N_LAYERS rows, 3 columns of subplots

plt.rcParams["font.family"] = "Times New Roman"

for i in range(1, N_LAYERS):
    fig.delaxes(axs[i, 2])

# Plotting head, layer, and average results in respective subplots
for i in range(N_LAYERS):
    # Head data

    # list possible cmap values: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    cmap = "coolwarm"
    vmax = 60

    sns.heatmap(np.array([head_data[i]]), annot=True, cmap=cmap, cbar=False, linewidths=0.0, ax=axs[i, 0], vmin=0, vmax=vmax, fmt='.1f')
    axs[i, 0].set_xticklabels([f"head {h+1}" for h in range(N_HEADS)], ha='center')

    # write horizontally
    axs[i, 0].set_yticklabels([f"layer {i+1}"], rotation=0, ha='right')
    axs[i, 0].tick_params(left=False, bottom=False)  # Remove y-ticks and x-ticks

    # remove x labels for all but the bottom row
    if i < N_LAYERS - 1:
        axs[i, 0].set_xticklabels([''] * N_HEADS, ha='center')
        axs[i, 0].tick_params(bottom=False)

    # Layer data
    sns.heatmap(np.array([layer_data[i]]), annot=True, cmap=cmap, cbar=False, linewidths=0.0, ax=axs[i, 1], vmin=0, vmax=vmax, fmt='.1f')
    axs[i, 1].set_xticklabels(['layer\naverage'], ha='center')
    axs[i, 1].set_yticklabels([''])
    axs[i, 1].tick_params(left=False, bottom=i == N_LAYERS - 1)  # Remove y-ticks and x-ticks

    if i < N_LAYERS - 1:
        axs[i, 1].set_xticklabels([''], ha='center')
        axs[i, 1].tick_params(bottom=False)

ax_big = fig.add_subplot(1, 3, 3)  # 1 row, 3 columns, position 3
sns.heatmap(average_data, annot=True, cmap=cmap, cbar=False, linewidths=0.0, ax=ax_big, vmin=0, vmax=vmax, fmt='.1f')
ax_big.set_xticklabels(['full\naverage'], ha='center')
ax_big.set_yticklabels([])
ax_big.tick_params(left=False, bottom=False)  # Remove y-ticks and x-ticks



# Adjust the layout
plt.tight_layout()

plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, hspace=0.2, wspace=0.1)


# adjust with of the big axis
pos1 = ax_big.get_position()  # get the original position
pos2 = [axs[0, 2].get_position().x0, pos1.y0, axs[0, 1].get_position().width, pos1.height]
ax_big.set_position(pos2)  # set a new position

# make axs[0, 2] invisible
axs[0, 2].set_visible(False)


plt.savefig("uuas_retrieval_base.png", dpi=300, bbox_inches='tight')
plt.savefig("uuas_retrieval_base.pdf", dpi=300, bbox_inches='tight')
plt.show()


