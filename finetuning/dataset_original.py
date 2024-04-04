from smart_open import open
import torch
from torch.utils.data import Dataset
from lemma_rule import gen_lemma_rule


class Dataset(Dataset):
    def __init__(self, path: str, partition: str, tokenizer, forms_vocab=None, lemma_vocab=None, upos_vocab=None, xpos_vocab=None, feats_vocab=None, arc_dep_vocab=None, add_sep=True, random_mask=False):
        entries, current = [], []
        for line in open(f'{path}-{partition}.conllu'):
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

        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.n_embeddings = 1
        self.pad_index = self.tokenizer.convert_tokens_to_ids("[PAD]")
        self.add_sep = add_sep
        self.random_mask = random_mask

        self.forms = [[current[1] for current in entry] for entry in entries]

        self.subwords, self.alignment = [], []
        n_splits, n = 0, 0
        for i_sentence, sentence in enumerate(self.forms):
            subwords, alignment = [self.tokenizer.convert_tokens_to_ids("[CLS]")], [0]
            for i, word in enumerate(sentence):
                space_before = (i == 0) or (not "SpaceAfter=No" in entries[i_sentence][i - 1][-1])

                # very very ugly hack ;(
                encoding = self.tokenizer(f"| {word}" if space_before else f"|{word}", add_special_tokens=False)
                subwords += encoding.input_ids[2:]
                alignment += (len(encoding.input_ids) - 2) * [i + 1]

                assert len(encoding.input_ids) > 2, f"{word} {encoding.input_ids}"
                # assert word == tokenizer.decode(encoding.input_ids[:]).strip(), f"{word} != {tokenizer.decode(encoding.input_ids[2:])}"

                if not word.isalpha():
                    continue
                n_splits += len(encoding.input_ids) - 2
                n += 1

            if self.add_sep:
                subwords.append(self.tokenizer.convert_tokens_to_ids("[SEP]"))
                alignment.append(alignment[-1] + 1)

            self.subwords.append(subwords)
            self.alignment.append(alignment)

        self.average_word_splits = n_splits / n
