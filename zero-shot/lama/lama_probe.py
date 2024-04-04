from transformers import AutoTokenizer, AutoModelForMaskedLM
from modeling_ltgbert import LtgBertForMaskedLM
import torch
import logging
import argparse
from tqdm import tqdm
import os
import numpy as np
import random
from cloze_dataset import ClozeDataset
from torch.utils.data import DataLoader
from typing import List, Dict

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Evaluate a PLM on the LAMA knowledge probe')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='The batch size to use during probing')
    parser.add_argument('--model_name', type=str,
                        default='ltg/bnc-bert-span', help='The pretrained model to use')
    parser.add_argument('--subset', type=str, default='Squad',
                        help='The subset of LAMA to probe on')
    parser.add_argument('--seed', type=int, default=42, help='The rng seed')
    parser.add_argument('--k', type=int, default=100, help='p @ k value')

    args = parser.parse_args()
    return args


def topk(prediction_scores, token_index, k=10, tokenizer=None) -> List:
    """
    :param predicition_scores: The logits for each word in the model's output vocabulary
    :param token_index: a list of the index of the mask token for each sample
    :param k: the number of top words to return
    :param tokenizer: the tokenizer

    :returns: The top k words for a masked token given a set of logits
    """
    preds_for_masks = prediction_scores[token_index]
    _, tops = torch.topk(input=preds_for_masks, k=k)
    top_words = tokenizer.convert_ids_to_tokens(tops)
    top_words = [t.strip('Ġ') for t in top_words]
    return top_words


def aggregate_metrics_elements(metrics_elements: List[Dict], k: int) -> Dict:
    """
    :param metrics_elements: A list of the individual measurements.
    :param k: the number of top words to return
    :returns: The aggregeated measurements
    """

    MRR = sum([x['MRR'] for x in metrics_elements]) / \
        len([x['MRR'] for x in metrics_elements])
    Precision10 = sum([x['P_AT_10'] for x in metrics_elements]) / \
        len([x['P_AT_10'] for x in metrics_elements])
    Precision1 = sum([x['P_AT_1'] for x in metrics_elements]) / \
        len([x['P_AT_1'] for x in metrics_elements])
    PrecisionK = sum([x[f'P_AT_K={k}'] for x in metrics_elements]) / \
        len([x[f'P_AT_K={k}'] for x in metrics_elements])

    aggregated = {
        'MRR': MRR,
        'P_AT_1': Precision1,
        'P_AT_10': Precision10,
        f'P_AT_K={k}': PrecisionK,
    }

    return aggregated


def probe(args, probing_model, tokenizer, data_loader) -> None:
    """
    :param args: CLI arguments.
    :param probing_model: The LM to probe
    :param tokenizer: The tokenizer to use
    :param data_loader: The data loader object holding the subset of LAMA

    :returns: The aggregeated measurements
    """

    precision_at_k = args.k
    metrics_elements = []

    for _, (input_ids, attention_masks, labels, mask_idx) in enumerate(tqdm(data_loader)):
        input_ids_batch = input_ids.to(device)
        attention_mask_batch = attention_masks.to(device)
        metrics_element = {}

        # Get predictions from models
        outputs = probing_model(
            input_ids_batch, attention_mask=attention_mask_batch).logits

        for i, prediction_scores in enumerate(outputs):
            metrics_element[f'P_AT_K={precision_at_k}'] = 0.0
            metrics_element['P_AT_10'] = 0.0
            metrics_element['P_AT_1'] = 0.0
            metrics_element['MRR'] = 0.0
            if mask_idx[i] == -1:
                continue  # No MASK found in tokenized dataset. See dataset class

            topk_tokens = topk(prediction_scores, mask_idx[i], k=precision_at_k, tokenizer=tokenizer)[
                :precision_at_k]
            
            #print(labels[i])
            #print(topk_tokens)

            try:
                rank = topk_tokens.index(labels[i])
                metrics_element['MRR'] = (1 / (rank + 1))
                if rank <= precision_at_k:
                    metrics_element[f'P_AT_K={precision_at_k}'] = 1
                if rank <= 10:
                    metrics_element['P_AT_10'] = 1
                if rank == 0:
                    metrics_element['P_AT_1'] = 1

            except:
                metrics_element['rank'] = f'not found in top {precision_at_k} words'

            metrics_elements.append(metrics_element)

    logging.info(f'Number metrics elements: {len(metrics_elements)}')
    aggregated_metrics = aggregate_metrics_elements(
        metrics_elements, k=precision_at_k)

    if precision_at_k <= 10:
        aggregated_metrics.pop('P_AT_10')

    logging.info(f'Aggregated: {aggregated_metrics}')


def main(args):
    logging.info('========== Loaded LAMA probe evaluation ==========')
    assert args.subset.lower() in ['squad', 'conceptnet', 'trex', 'google_re']
    test_dataset = ClozeDataset(
        '../data/lama_data/ConceptNet/test_special.jsonl', tokenizer_path_or_name=args.model_name)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    if "ltg" in args.model_name:
        probing_model = LtgBertForMaskedLM.from_pretrained(
            args.model_name).to(device)
    else:
        probing_model = AutoModelForMaskedLM.from_pretrained(
            args.model_name, trust_remote_code=True).to(device)
    logging.info(test_dataset.OOV_words)

    probe(args, probing_model, test_dataset.tokenizer, test_loader)


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main(args)
