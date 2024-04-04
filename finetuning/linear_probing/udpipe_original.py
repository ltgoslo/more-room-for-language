import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import dependency_decoding

import sys, os
from transformers import AutoModel


class Classifier(nn.Module):
    def __init__(self, hidden_size, vocab_size, dropout):
        super().__init__()

        self.transform = nn.Sequential(
#            nn.Linear(hidden_size, hidden_size),
#            nn.GELU(),
            nn.LayerNorm(hidden_size, elementwise_affine=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size)
        )
        self.initialize(hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
#        nn.init.trunc_normal_(self.transform[0].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.transform[-1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
#        self.transform[0].bias.data.zero_()
        self.transform[-1].bias.data.zero_()

    def forward(self, x):
        return self.transform(x)


class ZeroClassifier(nn.Module):
    def forward(self, x):
        output = torch.zeros(x.size(0), x.size(1), 2, device=x.device, dtype=x.dtype)
        output[:, :, 0] = 1.0
        output[:, :, 1] = -1.0
        return output


class EdgeClassifier(nn.Module):
    def __init__(self, hidden_size, dep_hidden_size, vocab_size, dropout):
        super().__init__()

        self.head_dep_transform = nn.Sequential(
#            nn.Linear(hidden_size, hidden_size),
#            nn.GELU(),
            nn.LayerNorm(hidden_size, elementwise_affine=False),
            nn.Dropout(dropout)
        )
        self.head_root_transform = nn.Sequential(
#            nn.Linear(hidden_size, hidden_size),
#            nn.GELU(),
            nn.LayerNorm(hidden_size, elementwise_affine=False),
            nn.Dropout(dropout)
        )
        self.head_bilinear = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.head_linear_dep = nn.Linear(hidden_size, 1, bias=False)
        self.head_linear_root = nn.Linear(hidden_size, 1, bias=False)
        self.head_bias = nn.Parameter(torch.zeros(1))

        dep_hidden_size = hidden_size

        self.dep_dep_transform = nn.Sequential(
#            nn.Linear(hidden_size, dep_hidden_size),
#            nn.GELU(),
            nn.LayerNorm(dep_hidden_size, elementwise_affine=False),
            nn.Dropout(dropout)
        )
        self.dep_root_transform = nn.Sequential(
#            nn.Linear(hidden_size, dep_hidden_size),
#            nn.GELU(),
            nn.LayerNorm(dep_hidden_size, elementwise_affine=False),
            nn.Dropout(dropout)
        )
        self.dep_bilinear = nn.Parameter(torch.zeros(dep_hidden_size, dep_hidden_size, vocab_size))
        self.dep_linear_dep = nn.Linear(dep_hidden_size, vocab_size, bias=False)
        self.dep_linear_root = nn.Linear(dep_hidden_size, vocab_size, bias=False)
        self.dep_bias = nn.Parameter(torch.zeros(vocab_size))

        self.hidden_size = hidden_size
        self.dep_hidden_size = dep_hidden_size

        self.mask_value = float("-inf")
        self.initialize(hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
 #       nn.init.trunc_normal_(self.head_dep_transform[0].weight, mean=0.0, std=std, a=-2*std, b=2*std)
 #       nn.init.trunc_normal_(self.head_root_transform[0].weight, mean=0.0, std=std, a=-2*std, b=2*std)
 #       nn.init.trunc_normal_(self.dep_dep_transform[0].weight, mean=0.0, std=std, a=-2*std, b=2*std)
 #       nn.init.trunc_normal_(self.dep_root_transform[0].weight, mean=0.0, std=std, a=-2*std, b=2*std)

        nn.init.trunc_normal_(self.head_linear_dep.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.head_linear_root.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.dep_linear_dep.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.dep_linear_root.weight, mean=0.0, std=std, a=-2*std, b=2*std)

 #       self.head_dep_transform[0].bias.data.zero_()
 #       self.head_root_transform[0].bias.data.zero_()
 #       self.dep_dep_transform[0].bias.data.zero_()
 #       self.dep_root_transform[0].bias.data.zero_()

    def forward(self, x, lengths, head_gold=None):
        head_dep = self.head_dep_transform(x[:, 1:, :])
        head_root = self.head_root_transform(x)
        head_prediction = torch.einsum("bkn,nm,blm->bkl", head_dep, self.head_bilinear, head_root / math.sqrt(self.hidden_size)) \
            + self.head_linear_dep(head_dep) + self.head_linear_root(head_root).transpose(1, 2) + self.head_bias

        mask = (torch.arange(x.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)).unsqueeze(1).to(x.device)
        mask = mask | (torch.ones(x.size(1) - 1, x.size(1), dtype=torch.bool, device=x.device).tril(1) & torch.ones(x.size(1) - 1, x.size(1), dtype=torch.bool, device=x.device).triu(1))
        head_prediction = head_prediction.masked_fill(mask, self.mask_value)

        if head_gold is None:
            head_logp = torch.log_softmax(head_prediction, dim=-1)
            head_logp = F.pad(head_logp, (0, 0, 1, 0), value=torch.nan).cpu()
            head_gold = []
            for i, length in enumerate(lengths.tolist()):
                local_logp = head_logp[i, :length, :length].clone()
                local_logp[:, 0] = torch.nan
                local_logp[1 + head_logp[i, 1:length, 0].argmax(), 0] = 0
                head, _ = dependency_decoding.chu_liu_edmonds(local_logp.numpy().astype(float))
                head = head[1:] + ((x.size(1) - 1) - (len(head) - 1)) * [0]
                head_gold.append(torch.tensor(head))
            head_gold = torch.stack(head_gold).to(x.device)

        dep_dep = self.dep_dep_transform(x[:, 1:])
        dep_root = x.gather(1, head_gold.unsqueeze(-1).expand(-1, -1, x.size(-1)).clamp(min=0))
        dep_root = self.dep_root_transform(dep_root)
        dep_prediction = torch.einsum("btm,mnl,btn->btl", dep_dep, self.dep_bilinear, dep_root / math.sqrt(self.dep_hidden_size)) \
            + self.dep_linear_dep(dep_dep) + self.dep_linear_root(dep_root) + self.dep_bias

        return head_prediction, dep_prediction, head_gold


class Model(nn.Module):
    def __init__(self, args, dataset, use_context: bool):
        super().__init__()

        self.bert = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
        if args.freeze:
            self.bert.requires_grad_(False)

        self.dropout = nn.Dropout(args.dropout)
        self.layer_norm = nn.LayerNorm(args.hidden_size, elementwise_affine=False)
        self.upos_layer_score = nn.Parameter(torch.zeros(args.num_layers+1, dtype=torch.float))
        self.xpos_layer_score = nn.Parameter(torch.zeros(args.num_layers+1, dtype=torch.float))
        self.feats_layer_score = nn.Parameter(torch.zeros(args.num_layers+1, dtype=torch.float))
        self.lemma_layer_score = nn.Parameter(torch.zeros(args.num_layers+1, dtype=torch.float))
        self.dep_layer_score = nn.Parameter(torch.zeros(args.num_layers+1, dtype=torch.float))

        self.lemma_classifier = Classifier(args.hidden_size, len(dataset.lemma_vocab), args.dropout)
        self.upos_classifier = Classifier(args.hidden_size, len(dataset.upos_vocab), args.dropout) if len(dataset.upos_vocab) > 2 else ZeroClassifier()
        self.xpos_classifier = Classifier(args.hidden_size, len(dataset.xpos_vocab), args.dropout) if len(dataset.xpos_vocab) > 2 else ZeroClassifier()
        self.feats_classifier = Classifier(args.hidden_size, len(dataset.feats_vocab), args.dropout) if len(dataset.feats_vocab) > 2 else ZeroClassifier()
        self.edge_classifier = EdgeClassifier(args.hidden_size, 128, len(dataset.arc_dep_vocab), args.dropout)

    def forward(self, x, alignment_mask, subword_lengths, word_lengths, head_gold=None):
        padding_mask = (torch.arange(x.size(1)).unsqueeze(0) < subword_lengths.unsqueeze(1)).to(x.device)
        x = self.bert(x, padding_mask, output_hidden_states=True, return_dict=True).hidden_states
        x = torch.stack(x, dim=0)

        upos_x = torch.einsum("lbtd, l -> btd", x, torch.softmax(self.upos_layer_score, dim=0))
        xpos_x = torch.einsum("lbtd, l -> btd", x, torch.softmax(self.xpos_layer_score, dim=0))
        feats_x = torch.einsum("lbtd, l -> btd", x, torch.softmax(self.feats_layer_score, dim=0))
        lemma_x = torch.einsum("lbtd, l -> btd", x, torch.softmax(self.lemma_layer_score, dim=0))
        dep_x = torch.einsum("lbtd, l -> btd", x, torch.softmax(self.dep_layer_score, dim=0))

        upos_x = torch.einsum("bsd,bst->btd", upos_x, alignment_mask) / alignment_mask.sum(1).unsqueeze(-1).clamp(min=1.0)
        xpos_x = torch.einsum("bsd,bst->btd", xpos_x, alignment_mask) / alignment_mask.sum(1).unsqueeze(-1).clamp(min=1.0)
        feats_x = torch.einsum("bsd,bst->btd", feats_x, alignment_mask) / alignment_mask.sum(1).unsqueeze(-1).clamp(min=1.0)
        lemma_x = torch.einsum("bsd,bst->btd", lemma_x, alignment_mask) / alignment_mask.sum(1).unsqueeze(-1).clamp(min=1.0)
        dep_x = torch.einsum("bsd,bst->btd", dep_x, alignment_mask) / alignment_mask.sum(1).unsqueeze(-1).clamp(min=1.0)

        upos_x = self.dropout(self.layer_norm(upos_x[:, 1:-1, :]))
        xpos_x = self.dropout(self.layer_norm(xpos_x[:, 1:-1, :]))
        feats_x = self.dropout(self.layer_norm(feats_x[:, 1:-1, :]))
        lemma_x = self.dropout(self.layer_norm(lemma_x[:, 1:-1, :]))
        dep_x = self.dropout(self.layer_norm(dep_x[:, 0:-1, :]))

        lemma_preds = self.lemma_classifier(lemma_x)
        upos_preds = self.upos_classifier(upos_x)
        xpos_preds = self.xpos_classifier(xpos_x)
        feats_preds = self.feats_classifier(feats_x)
        head_prediction, dep_prediction, head_liu = self.edge_classifier(dep_x, word_lengths, head_gold)

        return lemma_preds, upos_preds, xpos_preds, feats_preds, head_prediction, dep_prediction, head_liu
