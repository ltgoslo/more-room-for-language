import torch
import torch.nn.functional as F


@torch.no_grad()
def rank(sentences, model, tokenizer, device, batch_size, temperatures=None):
    mask_index = tokenizer.mask_token_id
    pad_index = tokenizer.pad_token_id
    cls_index = torch.tensor([tokenizer.cls_token_id])
    sep_index = torch.tensor([tokenizer.sep_token_id])

    sentences = [tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0) for s in sentences]
    labels = torch.cat(sentences).unsqueeze(-1).expand(temperatures.size(0), -1, -1).to(device)

    if temperatures is None:
        temperatures = torch.ones(1, device=device)

    def prepare(tokens, padding: int):
        tokens = torch.cat([cls_index, tokens, sep_index, torch.full((padding,), fill_value=pad_index)]).to(device)
        tokens = tokens.repeat(tokens.size(0) - 2 - padding, 1)
        mask = torch.eye(tokens.size(1), device=device).bool()[1:-(1 + padding), :]
        input_ids = tokens.masked_fill(mask, value=mask_index)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        attention_mask[:, attention_mask.size(-1) - padding:] = False
        return input_ids, attention_mask

    max_length = max(s.size(0) for s in sentences)
    input_ids, attention_masks = zip(*[prepare(s, max_length - s.size(0)) for s in sentences])

    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_masks, dim=0)

    indices = [torch.arange(1, 1 + len(s), device=device) for s in sentences]
    indices = torch.cat(indices, dim=0)

    total_score = []

    for b in range(input_ids.size(0) // batch_size + 1):
        logits = model(
            input_ids[b * batch_size : (b+1) * batch_size, :].contiguous(),
            attention_mask[b * batch_size : (b+1) * batch_size, :].contiguous()
        ).logits

        logits = torch.gather(
            logits,
            dim=1,
            index=indices[b * batch_size : (b+1) * batch_size].reshape(-1, 1, 1).expand(-1, -1, logits.size(-1))
        ).squeeze(1)
        logits = logits.unsqueeze(0) / temperatures.view(-1, 1, 1)
        log_p = F.log_softmax(logits, dim=-1)
        log_p = log_p.gather(index=labels[:, b * batch_size : (b+1) * batch_size, :], dim=-1).squeeze(-1)
        total_score.append(log_p)

    total_score = torch.cat(total_score, dim=1)

    log_ps, offset = [], 0
    for i in range(len(sentences)):
        from_index = offset
        to_index = offset + sentences[i].size(0)
        log_ps.append(total_score[:, from_index:to_index].sum(-1))
        offset = to_index

    ranking = torch.argsort(torch.stack(log_ps, dim=1), dim=1, descending=True).tolist()
    return ranking[int(1.0 / 0.05)], ranking
