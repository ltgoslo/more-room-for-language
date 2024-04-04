import tqdm
import wandb
import argparse
import random
import math
import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from conll18_ud_eval import evaluate, load_conllu_file
from dataset_original import Dataset
from udpipe_original import Model
from transformers import AutoTokenizer
from lemma_rule import apply_lemma_rule

torch.backends.cuda.matmul.allow_tf32 = True


def seed_everything(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)


class CrossEntropySmoothingMasked:
    def __init__(self, smoothing=0.0):
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def __call__(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=1)
        nll_loss = -logprobs.gather(dim=1, index=target.unsqueeze(1).clamp(min=0)).squeeze(1)

        logprobs = logprobs.masked_fill(x == float("-inf"), 0.0)
        smooth_loss = -logprobs.mean(dim=1)

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        loss = loss.masked_fill(target == -1, 0.0).sum() / (target != -1).float().sum()
        return loss


class CollateFunctor:
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, sentences):
        longest_source = max([sentence["subwords"].size(0) for sentence in sentences])
        longest_target = max([sentence["upos"].size(0) for sentence in sentences])

        return {
            "index": [sentence["index"] for sentence in sentences],
            "subwords": torch.stack([F.pad(sentence["subwords"], (0, longest_source - sentence["subwords"].size(0)), value=self.pad_index) for sentence in sentences]),
            "alignment": torch.stack(
                [
                    F.pad(F.one_hot(sentence["alignment"], num_classes=longest_target + 2).float(), (0, 0, 0, longest_source - sentence["alignment"].size(0)), value=0.0)
                    for sentence in sentences
                ]
            ),
            "is_unseen": torch.stack([F.pad(sentence["is_unseen"], (0, longest_target - sentence["is_unseen"].size(0)), value=False) for sentence in sentences]),
            "lemma": torch.stack([F.pad(sentence["lemma"], (0, longest_target - sentence["lemma"].size(0)), value=-1) for sentence in sentences]),
            "upos": torch.stack([F.pad(sentence["upos"], (0, longest_target - sentence["upos"].size(0)), value=-1) for sentence in sentences]),
            "xpos": torch.stack([F.pad(sentence["xpos"], (0, longest_target - sentence["xpos"].size(0)), value=-1) for sentence in sentences]),
            "feats": torch.stack([F.pad(sentence["feats"], (0, longest_target - sentence["feats"].size(0)), value=-1) for sentence in sentences]),
            "arc_head": torch.stack([F.pad(sentence["arc_head"], (0, longest_target - sentence["arc_head"].size(0)), value=-1) for sentence in sentences]),
            "arc_dep": torch.stack([F.pad(sentence["arc_dep"], (0, longest_target - sentence["arc_dep"].size(0)), value=-1) for sentence in sentences]),
            "subword_lengths": torch.LongTensor([sentence["subwords"].size(0) for sentence in sentences]),
            "word_lengths": torch.LongTensor([sentence["upos"].size(0) + 1 for sentence in sentences])
        }


def main():
    # load your own model
    parser = ArgumentParser()
    parser.add_argument("--freeze", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", default="base_paraphrase_patched_frozen")
    parser.add_argument("--model_path", default="pretrain/hugging_models/base_paraphrase_patched")
    parser.add_argument("--hidden_size", action="store", type=int, default=768)
    parser.add_argument("--num_layers", action="store", type=int, default=12)
    parser.add_argument("--batch_size", action="store", type=int, default=32)
    parser.add_argument("--lr", action="store", type=float, default=0.001)
    parser.add_argument("--beta", action="store", type=float, default=0.0)
    parser.add_argument("--alpha", action="store", type=float, default=0.1)
    parser.add_argument("--weight_decay", action="store", type=float, default=0.001)
    parser.add_argument("--dropout", action="store", type=float, default=0.2)
    parser.add_argument("--word_dropout", action="store", type=float, default=0.0)
    parser.add_argument("--label_smoothing", action="store", type=float, default=0.1)
    parser.add_argument("--epochs", action="store", type=int, default=10)
    parser.add_argument("--num_warmup_steps", action="store", type=int, default=250)
    parser.add_argument("--seed", action="store", type=int, default=42)
    parser.add_argument("--dataset", action="store", type=str, default="en_ewt-ud")
    parser.add_argument("--language", action="store", type=str, default="english")
    parser.add_argument("--tokenizer", action="store", type=str, default="clean")
    parser.add_argument("--bpe_subtype", action="store", type=str, default="")
    parser.add_argument("--log_wandb", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    seed_everything(args.seed)

    if args.log_wandb:
        wandb.init(name=f"{args.model}_{args.dataset}", config=args, project="PROJECT_NAME", entity="USERNAME")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    train_data = Dataset(f"data/{args.dataset}", partition='train', tokenizer=tokenizer, add_sep=True, random_mask=True)
    test_data = Dataset(
        f"data/{args.dataset}",
        partition='test',
        tokenizer=tokenizer,
        forms_vocab=train_data.forms_vocab,
        lemma_vocab=train_data.lemma_vocab,
        upos_vocab=train_data.upos_vocab,
        xpos_vocab=train_data.xpos_vocab,
        feats_vocab=train_data.feats_vocab,
        arc_dep_vocab=train_data.arc_dep_vocab,
        add_sep=True,
        random_mask=False
    )

    # build and pad with loaders
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True, num_workers=7, collate_fn=CollateFunctor(train_data.pad_index))
    test_loader = DataLoader(test_data, args.batch_size, shuffle=False, drop_last=False, num_workers=7, collate_fn=CollateFunctor(train_data.pad_index))

    model = Model(args, train_data, use_context=True).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.log_wandb:
        wandb.config.update({"params": n_params})
    print(f"{args.language}_{args.tokenizer}: {n_params}", flush=True)

    criterion = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=args.label_smoothing).to(device)
    masked_criterion = CrossEntropySmoothingMasked(args.label_smoothing)

    params = list(model.named_parameters())
    no_decay = {'bias', 'layer_norm', 'vectors', '_embedding', 'layer_score'}
    bert_decay_params = [(n, p) for n, p in params if not any(nd in n for nd in no_decay) and "bert" in n and p.requires_grad]
    bert_no_decay_params = [(n, p) for n, p in params if any(nd in n for nd in no_decay) and "bert" in n and p.requires_grad]
    decay_params = [(n, p) for n, p in params if not any(nd in n for nd in no_decay) and not "bert" in n and p.requires_grad]
    no_decay_params = [(n, p) for n, p in params if any(nd in n for nd in no_decay) and not "bert" in n and p.requires_grad]
    optimizer_grouped_parameters = [
        {'params': [p for _, p in bert_decay_params], 'lr': 0.1*args.lr, 'weight_decay': 0.1},
        {'params': [p for _, p in bert_no_decay_params], 'lr': 0.1*args.lr, 'weight_decay': 0.0},
        {'params': [p for _, p in decay_params], 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': [p for _, p in no_decay_params], 'lr': args.lr, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.99))

    def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_factor: float):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(min_factor, min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler = cosine_schedule_with_warmup(optimizer, args.num_warmup_steps, args.epochs * len(train_loader), 0.1)

    # train loop
    global_step = 1
    for epoch in range(args.epochs):
        train_iter = tqdm.tqdm(train_loader)
        model.train()
        for batch in train_iter:
            optimizer.zero_grad(set_to_none=True)

            lemma_p, upos_p, xpos_p, feats_p, head_p, dep_p, _ = model(
                batch["subwords"].to(device),
                batch["alignment"].to(device),
                batch["subword_lengths"],
                batch["word_lengths"],
                batch["arc_head"].to(device)
            )

            lemma_loss = criterion(lemma_p.transpose(1, 2), batch["lemma"].to(device))
            upos_loss = criterion(upos_p.transpose(1, 2), batch["upos"].to(device))
            xpos_loss = criterion(xpos_p.transpose(1, 2), batch["xpos"].to(device))
            feats_loss = criterion(feats_p.transpose(1, 2), batch["feats"].to(device))
            head_loss = masked_criterion(head_p.transpose(1, 2), batch["arc_head"].to(device))
            dep_loss = criterion(dep_p.transpose(1, 2), batch["arc_dep"].to(device))

            loss = 0 * lemma_loss + 0 * upos_loss + 0 * xpos_loss + 0 * feats_loss + head_loss + dep_loss
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()

            global_step += 1

            if args.log_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/lemma_loss": lemma_loss.item(),
                        "train/upos_loss": upos_loss.item(),
                        "train/xpos_loss": xpos_loss.item(),
                        "train/feats_loss": feats_loss.item(),
                        "train/head_loss": head_loss.item(),
                        "train/dep_loss": dep_loss.item(),
                        "train/loss": loss.item(),
                        "stats/grad_norm": grad_norm.item(),
                        "stats/learning_rate": optimizer.param_groups[0]['lr'],
                    }
                )
            train_iter.set_postfix_str(f"loss: {loss.item()}")

        # eval
        with torch.no_grad():
            model.eval()
            with open(f"{args.model}_{args.seed}.test.conllu", "w") as f:
                for batch in test_loader:
                    lemma_p, upos_p, xpos_p, feats_p, _, dep_p, head_p = model(
                        batch["subwords"].to(device),
                        batch["alignment"].to(device),
                        batch["subword_lengths"],
                        batch["word_lengths"],
                    )

                    for i, index in enumerate(batch["index"]):
                        for j, form in enumerate(test_data.forms[index]):
                            lemma_rule = test_data.lemma_vocab[lemma_p[i, j, :].argmax().item()]
                            lemma = apply_lemma_rule(form, lemma_rule)
                            upos = test_data.upos_vocab[upos_p[i, j, :].argmax().item()]
                            xpos = test_data.xpos_vocab[xpos_p[i, j, :].argmax().item()]
                            feats = test_data.feats_vocab[feats_p[i, j, :].argmax().item()]
                            head = head_p[i, j].item()
                            dep = test_data.arc_dep_vocab[dep_p[i, j, :].argmax().item()]
                            suffix = "UNSEEN" if batch["is_unseen"][i, j].item() else "_"
                            f.write(f"{j+1}\t{form}\t{lemma}\t{upos}\t{xpos}\t{feats}\t{head}\t{dep}\t{head}:{dep}\t{suffix}\n")
                        f.write("\n")

        try:
            gold_ud = load_conllu_file(f"data/{args.dataset}-test.conllu")
            system_ud = load_conllu_file(f"{args.model}_{args.seed}.test.conllu")
            evaluation = evaluate(gold_ud, system_ud)

#    if args.log_wandb:
#        wandb.log(
#                    {
#                        "epoch": epoch,
#                        "valid/UPOS": evaluation["UPOS"].aligned_accuracy * 100,
#                        "valid/XPOS": evaluation["XPOS"].aligned_accuracy * 100,
#                        "valid/UFeats": evaluation["UFeats"].aligned_accuracy * 100,
#                        "valid/AllTags": evaluation["AllTags"].aligned_accuracy * 100,
#                        "valid/Lemmas": evaluation["Lemmas"].aligned_accuracy * 100,
#                        "valid/UAS": evaluation["UAS"].aligned_accuracy * 100,
#                        "valid/LAS": evaluation["LAS"].aligned_accuracy * 100,
#                        "valid/MLAS": evaluation["MLAS"].aligned_accuracy * 100,
#                        "valid/CLAS": evaluation["CLAS"].aligned_accuracy * 100,
#                        "valid/BLEX": evaluation["BLEX"].aligned_accuracy * 100
#                    }
#                )

        except:
            pass

    with open(f"results_{args.model}_test_linear_las_only.txt", 'a') as f:
        values = [args.seed] + [evaluation[k].aligned_accuracy * 100 for k in ["UPOS", "XPOS", "UFeats", "AllTags", "Lemmas", "UAS", "LAS", "CLAS", "MLAS", "BLEX"]]
        f.write(','.join([str(v) for v in values]) + "\n")


if __name__ == '__main__':
    main()
