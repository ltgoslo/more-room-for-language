import torch
from torch.utils.data import Dataset
import json
import pandas as pd
from transformers import AutoTokenizer
import numpy as np


class ClozeDataset(Dataset):
    def __init__(self, path_to_dataset: str, tokenizer_path_or_name='davda54/RER_no-retrieval', max_length=64):
        trex_mode = True if "trex" in path_to_dataset.lower() else False
        df = pd.read_json(path_to_dataset, lines=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_name)
        vocab = self.tokenizer.get_vocab()

        if "trex" in path_to_dataset.lower():
            masked_sentences = []
            for index, row in df.iterrows():
                sentence = row["evidences"][0]["masked_sentence"]
                label = row["evidences"][0]["obj_surface"]
                masked_sentences.append(sentence)

            df["masked_sentences"] = masked_sentences

        self.labels = []
        texts = []
        self.OOV_words = 0
        for label, text in zip(df["obj_label"], df["masked_sentences"]):  # Skip OOV words
            # Important note: As in Petroni et al (2019), we limit the object labels to those exisiting in the intersection of all evaluated models vocabulary.
            if label in vocab.keys() or 'Ġ' + label in vocab.keys():
                self.labels.append(label)
                t = text[0] if not trex_mode else text
                texts.append(t)
            else:
                print(label)
                self.OOV_words += 1

        self.features = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True
        )

        self.mask_idx = []
        for i, x in enumerate(self.features["input_ids"]):
            try:
                self.mask_idx.append(x.index(self.tokenizer.mask_token_id))
            except:
                self.mask_idx.append(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.LongTensor(self.features["input_ids"][index]), torch.BoolTensor(self.features["attention_mask"][index]), self.labels[index], self.mask_idx[index]

    def get_decoded_example(self, index) -> str:
        return f'EXAMPLE: {self.tokenizer.decode(self.features["input_ids"][index])}, LABEL: {self.labels[index]}'


if __name__ == '__main__':
    set = ClozeDataset("../data/lama_data/Squad/test_special.jsonl")
    print(set[3])
    #for i in range(0, 20):
        #print(set.get_decoded_example(i))
