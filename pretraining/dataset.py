import torch
import json


class SpanMaskingStrategy:
    def __init__(self, mask_p, tokenizer, n_special_tokens, padding_label_id=-100, random_p=0.1, keep_p=0.1):
        self.mask_p = mask_p
        self.random_p = random_p
        self.keep_p = keep_p
        self.tokenizer = tokenizer
        self.n_special_tokens = n_special_tokens
        self.padding_label_id = padding_label_id
        self.mask_index = self.tokenizer.token_to_id("[MASK]")

    def __call__(self, tokens):
        labels = torch.full_like(tokens, fill_value=self.padding_label_id)
        inputs = tokens.clone()

        n_masked = torch.binomial((tokens >= self.n_special_tokens).float().sum(dim=0, keepdim=True), torch.FloatTensor([self.mask_p])).item()
        n_masked = min((tokens >= self.n_special_tokens).long().sum(dim=0), max(1, n_masked))
        preservation_mask = tokens < self.n_special_tokens
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        counter = 100

        while n_masked > mask.long().sum() and counter > 0:
            span_length = torch.tensor([0]).geometric_(1/3).item() % 10
            if span_length == 0:
                continue
            offset = torch.randint(-(span_length - 1), tokens.size(0) + span_length, []).item()
            sub_mask = torch.zeros_like(tokens, dtype=torch.bool)
            sub_mask[max(0, offset) : min(mask.size(0)-1, offset + span_length)] = True
            sub_mask[preservation_mask] = False

            random_p = torch.rand([]).item()

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            if random_p < 1.0 - self.random_p - self.keep_p:
                inputs[sub_mask] = self.mask_index
            elif random_p < 1.0 - self.keep_p:
                random_words = torch.randint(
                    low=self.n_special_tokens - 1,
                    high=self.tokenizer.get_vocab_size(),
                    size=(sub_mask.sum(),),
                    dtype=torch.long
                )
                inputs[sub_mask] = random_words
            else:
                inputs[sub_mask] = tokens[sub_mask]

            mask |= sub_mask
            counter -= 1

        labels[mask] = tokens[mask]

        return inputs, labels


class SpanMaskingStrategy2:
    def __init__(self, mask_p, tokenizer, n_special_tokens, padding_label_id=-100, random_p=0.1, keep_p=0.1):
        self.mask_p = mask_p
        self.random_p = random_p
        self.keep_p = keep_p
        self.vocab_size = tokenizer.get_vocab_size()
        self.n_special_tokens = n_special_tokens
        self.padding_label_id = padding_label_id
        self.mask_token_id = tokenizer.token_to_id("[MASK]")

    def __call__(self, tokens):

        replacement_tokens = tokens.clone()
        length = tokens.size(0)

        preservation_mask = tokens < self.n_special_tokens

        span_lengths = torch.zeros([length // 2]).geometric_(0.2) % 11
        span_lengths = span_lengths.clamp(1, 10).long()
        span_random_numbers_1 = torch.rand([length // 2])
        span_random_numbers_2 = torch.rand([length // 2])

        indices = torch.repeat_interleave(torch.arange(span_lengths.size(0)), span_lengths)
        indices = indices[:length]
        if indices.size(0) < length:
            indices = torch.cat([indices, torch.full([length - indices.size(0)], fill_value=length // 2 - 1, dtype=torch.long)])
        
        mask_ratios = span_random_numbers_1[indices]
        mask_ratios[preservation_mask] = 1.0

        replacement_p = span_random_numbers_2[indices]
        random_mask = replacement_p < self.random_p
        replacement_tokens[random_mask] = torch.randint(
            low=self.n_special_tokens,
            high=self.vocab_size,
            size=[random_mask.sum().item()],
            dtype=torch.long
        )
        replacement_tokens[replacement_p > (self.random_p + self.keep_p)] = self.mask_token_id

        mask = mask_ratios < self.mask_p
        target_ids = torch.where(mask, tokens, -100)
        input_ids = torch.where(mask, replacement_tokens, tokens)

        real_mask_p = mask.sum().item() / mask_ratios.numel()

        return input_ids, target_ids

class Dataset(torch.utils.data.Dataset):
    def __init__(self, gpu_id, n_gpus, tokenizer, seq_length=128, mask_p=0.15, short_p=0.1, random_p=0.1, keep_p=0.1, noisy_context_ratio=0.0):
        self.tokenizer = tokenizer

        self.seq_length = seq_length
        self.short_p = short_p
        self.n_special_tokens = 6
        self.noisy_context_ratio = noisy_context_ratio

        self.masking_strategy = SpanMaskingStrategy2(mask_p, tokenizer, self.n_special_tokens, padding_label_id=-100, random_p=random_p, keep_p=keep_p)

        self.mask_index = self.tokenizer.token_to_id("[MASK]")
        self.cls_index = self.tokenizer.token_to_id("[CLS]")
        self.sep_index = self.tokenizer.token_to_id("[SEP]")
        self.pad_index = self.tokenizer.token_to_id("[PAD]")

        self.original_segments = self.load("/scratch/project_465000144/dasamuel/RER/data/original.jsonl", n_gpus, gpu_id, tokenizer)
        self.paraphrased_segments = self.load("/scratch/project_465000144/dasamuel/RER/data/paraphrased.jsonl", n_gpus, gpu_id, tokenizer)

    def load(self, input_file, n_gpus, gpu_id, tokenizer):
        documents = [json.loads(line) for line in open(input_file, "r")]
        documents = documents[gpu_id:len(documents) // n_gpus * n_gpus:n_gpus]
        segments = [
            tokenizer.encode(document.strip(), add_special_tokens=False).ids
            for document in documents
        ]
        return segments

    def __len__(self):
        return len(self.original_segments)

    def __getitem__(self, index):
        original_tokens = self.original_segments[index]

        if torch.rand([]).item() < self.noisy_context_ratio:
            random_index = torch.randint(0, len(self.original_segments), []).item()
            paraphrased_tokens = self.paraphrased_segments[random_index]
        else:
            paraphrased_tokens = self.paraphrased_segments[index]

        original_tokens = original_tokens[:self.seq_length - 2]
        paraphrased_tokens = paraphrased_tokens[:self.seq_length - 2 + 64]

        original_padding_length = (self.seq_length - 2) - len(original_tokens)
        original_tensor = [self.cls_index] + original_tokens + [self.sep_index] + [self.pad_index] * original_padding_length
        original_tensor = torch.LongTensor(original_tensor)

        paraphrased_padding_length = (self.seq_length - 2 + 64) - len(paraphrased_tokens)
        paraphrased_tensor = [self.cls_index] + paraphrased_tokens + [self.sep_index] + [self.pad_index] * paraphrased_padding_length
        paraphrased_tensor = torch.LongTensor(paraphrased_tensor)
        
        original_attention_mask = torch.cat([
            torch.zeros(len(original_tokens) + 2, dtype=torch.bool),
            torch.ones(original_padding_length, dtype=torch.bool)
        ])
        paraphrased_attention_mask = torch.cat([
            torch.zeros(len(paraphrased_tokens) + 2, dtype=torch.bool),
            torch.ones(paraphrased_padding_length, dtype=torch.bool)
        ])

        original_inputs, original_outputs = self.masking_strategy(original_tensor)

        return original_inputs, original_outputs, original_attention_mask, paraphrased_tensor, paraphrased_attention_mask
    