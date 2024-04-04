import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import wandb
import numpy as np
import torch
import random
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, roc_auc_score
from tqdm import tqdm
import torch.nn.functional as F
import os

def linear_schedule_with_warmup(optimizer, warmup_ratio, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step: int):
        num_warmup_steps = int(warmup_ratio * num_training_steps)
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def seed_everything(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Details for msgs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='The batch size to use during finetuning')
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Maximum learning rate")
    parser.add_argument('--model_name', type=str,
                        default='davda54/wiki-retrieval-patch-base', help='The pretrained model to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='The RNG seed')
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight Decay to apply to the AdamW optimizer.")
    parser.add_argument("--data_dir", type=str, default="./data/datasets",
                        help="Location of the directory containing the MSGS dataset.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Maximum sequence length.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of fine-tuning epochs.")
    parser.add_argument("--warmup_ratio", type=float, default=0.06,
                        help="Percentage of warmup steps.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Value of epsilon for the AdamW optimizer.")
    parser.add_argument("--test_split", type=str, default="out_unmixed",
                        "The test split to use")



    args = parser.parse_args()
    return args

class MSGSDataset(Dataset):
    def __init__(self, file_path):
        
        with open(file_path, 'r') as json_file:
            json_list = list(json_file)
        
        first_rec = json.loads(json_list[0])
        
        if first_rec["control_paradigm"]:
            if first_rec["surface_feature_label"] is None:
                labeller = "linguistic_feature_label"
            else:
                labeller = "surface_feature_label"
        else:
            labeller = "linguistic_feature_label"
            
        self.texts = []
        self.labels = []
        
        for json_str in json_list:
            datapoint = json.loads(json_str)
            self.texts.append(datapoint["sentence"])
            self.labels.append(datapoint[labeller])
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], torch.tensor(self.labels[idx], dtype=torch.float)


if __name__ == "__main__":
    
    args = parse_args()

    root = args.data_dir
    dirlist = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]

    for d in dirlist:
        args.task = d
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        args.wandb_id = wandb.util.generate_id()

        if d.endswith("control"):
            test_split = "out"
            lab = ["Failed", "Passed"]
        else:
            test_split = args.test_split
            lab = ["Surface", "Linguistic"]
        print(f"{args.task}_{args.learning_rate}_{args.seed}_{args.model_name.split('/')[-1]}_{test_split}")

        wandb.init(
                name=f"{args.task}_{args.learning_rate}_{args.seed}_{args.model_name.split('/')[-1]}_{test_split}",
                config=args,
                id=args.wandb_id,
                project="PROJECT_NAME",
                entity="USERNAME",
                resume="auto",
                allow_val_change=True,
                reinit=True,
            )

        seed_everything(args.seed)

        train_set = MSGSDataset(args.data_dir+"/"+args.task+"/train.jsonl")
        test_set = MSGSDataset(args.data_dir+"/"+args.task+"/"+test_split+".jsonl")

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size)

        t_total = args.epochs * len(train_loader)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, trust_remote_code=True, num_labels=1).to(device)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = linear_schedule_with_warmup(
            optimizer, warmup_ratio=args.warmup_ratio, num_training_steps=t_total
            )

        progress_bar = tqdm(total=t_total)

        for i in range(args.epochs):
            model.train()
            for texts, labels in train_loader:
                inputs = tokenizer(texts, padding=True, max_length=args.seq_length, truncation=True, return_tensors="pt").to(device)
                labels = labels.to(device)
                output = model(inputs["input_ids"], inputs["attention_mask"]).logits
                preds = F.sigmoid(output)
                preds = torch.round(preds)
                loss = F.binary_cross_entropy_with_logits(output, labels.view(-1, 1))
                mcc = matthews_corrcoef(labels.detach().cpu().numpy(), preds.squeeze().detach().cpu().numpy())
                f1 = f1_score(labels.detach().cpu().numpy(), preds.squeeze().detach().cpu().numpy(), average="binary")
                f1_0 = f1_score(labels.detach().cpu().numpy(), preds.squeeze().detach().cpu().numpy(), pos_label=0, average="binary")
                acc = accuracy_score(labels.detach().cpu().numpy(), preds.squeeze().detach().cpu().numpy())
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update()
                progress_bar.set_postfix_str(f"Loss: {loss.item():.4f} MCC: {mcc:.2f}")
                wandb.log({
                    "train/Loss": loss.item(),
                    "train/MCC": mcc,
                    "train/F1": f1,
                    "train/F1_0": f1_0,
                    "train/Accuracy": acc,
                })

            progress_bar2 = tqdm(total=len(test_loader))

            model.eval()
            preds = []
            soft_logits = []
            total_loss = 0.0
            for texts, labels in test_loader:
                inputs = tokenizer(texts, padding=True, max_length=args.seq_length, truncation=True, return_tensors="pt").to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    output = model(inputs["input_ids"], inputs["attention_mask"]).logits
                soft_logit = F.sigmoid(output)
                pred = torch.round(soft_logit)
                preds.append(pred.detach().cpu().squeeze())
                soft_logits.append(soft_logit.detach().cpu().squeeze())
                loss = F.binary_cross_entropy_with_logits(output, labels.view(-1, 1))
                total_loss += loss
                mcc = matthews_corrcoef(labels.detach().cpu().numpy(), pred.squeeze().detach().cpu().numpy())
                progress_bar2.update()
                progress_bar2.set_postfix_str(f"Loss: {loss.item():.4f} MCC: {mcc:.2f}")
            
            progress_bar2.close()

            total_loss /= len(test_loader)
            preds = torch.cat(preds)
            soft_logits = torch.cat(soft_logits)
            eval_mcc = matthews_corrcoef(np.array(test_set.labels), preds.numpy())
            eval_f1 = f1_score(np.array(test_set.labels), preds.numpy(), average="binary")
            eval_f1_0 = f1_score(np.array(test_set.labels), preds.numpy(), pos_label=0, average="binary")
            eval_acc = accuracy_score(np.array(test_set.labels), preds.numpy())
            wandb.log({
                "test/Loss": total_loss,
                "test/MCC": eval_mcc,
                "test/F1": eval_f1,
                "test/F1_0": eval_f1_0,
                "test/Accuracy": eval_acc,
            })

        soft_logits2 = torch.cat([(1 - soft_logits).unsqueeze(-1), soft_logits.unsqueeze(-1)], dim=1)
        wandb.log({
            "results/Loss": total_loss,
            "results/MCC": eval_mcc,
            "results/F1": eval_f1,
            "results/F1_0": eval_f1_0,
            "results/Accuracy": eval_acc,
            "results/AUC": roc_auc_score(np.array(test_set.labels), soft_logits.numpy())
        })
        progress_bar.close()

        with open(f"results/{args.model_name.split('/')[-1]}__{args.seed}__{args.learning_rate}__{args.test_split}.txt", "a") as f:
            f.write(f"{args.task}: {eval_mcc}\n")

        wandb.finish()

