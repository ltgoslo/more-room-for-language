import torch
from model import load_model, paraphrase
import argparse
import json
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--input_file", type=str, default="/scratch/project_465000144/corpora/en_wiki_2023/segmented_clean_128_for_paraphrase_shorter.jsonl")
    parser.add_argument("--output_file_paraphrased", type=str, default="../data/paraphrased_{0}.jsonl")
    parser.add_argument("--output_file_original", type=str, default="../data/original_{0}.jsonl")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--worker_id", type=int, default=0)
    return parser.parse_args()
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    model, tokenizer = load_model(device, args.model_id)

    with open(args.input_file, "rt") as f:
        documents = [json.loads(line) for line in f]
        documents = [document for document in tqdm(documents) if len(document) > 0 and len(document.strip()) > 0]
        documents = documents[args.worker_id::128]

    # sort by length
    doc_token_lens = [len(tokenizer.encode(document)) for document in tqdm(documents)]
    documents = [document for _, document in sorted(zip(doc_token_lens, documents))][1024:-1024][::-1]
    del doc_token_lens

    print(f"Loaded {len(documents)} documents", flush=True)

    with open(args.output_file_original.format(args.worker_id), "wt") as f:
        for document in tqdm(documents):
            f.write(json.dumps(document.strip()) + "\n")

    with open(args.output_file_paraphrased.format(args.worker_id), "wt") as f:
        for i in tqdm(range(0, len(documents), args.batch_size)):
            batch = documents[i:i+args.batch_size]
            paraphrases = paraphrase(model, tokenizer, batch, device)
            for p in paraphrases:
                f.write(json.dumps(p.strip()) + "\n")
            f.flush()
