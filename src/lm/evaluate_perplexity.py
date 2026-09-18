import argparse
import json
import math
from einops import rearrange

import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm.utils import determine_device, enable_tf32
from lm.use_pythia import initialize_pythia


@torch.inference_mode()
def compute_perplexity(
    model: AutoModelForCausalLM,
    device: str,
    tokenizer: AutoTokenizer,
    documents: list[str],
    batch_size: int,
) -> float:
    """Computes perplexity given a list of documents

    Args:
        model: the language model
        device: device to put the tensors on
        tokenizer: the tokenizer
        documents: a list of document strings
        batch_size: number of documents to batch together during generation

    Returns:
        perplexity: a floating point number
    """

    total_loss = 0.0
    total_tokens = 0

    # do batched generation
    for i in trange(0, len(documents), batch_size):
        # tokenize the prefixes
        batch_docs = documents[i : i + batch_size]
        tokenized_docs = tokenizer(batch_docs, return_tensors="pt", padding=True)

        iids = tokenized_docs["input_ids"].to(device)
        mask = tokenized_docs["attention_mask"].to(device)

        logits = model(input_ids=iids, attention_mask=mask).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = iids[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous()

        # Unreduced loss per token
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.size())

        masked_loss = loss * shift_mask

        # --- Added: Calculate and print per-document perplexity ---
        doc_losses = masked_loss.sum(dim=1)
        doc_tokens = shift_mask.sum(dim=1)
        for doc_idx, (d_loss, d_tokens) in enumerate(zip(doc_losses, doc_tokens)):
            global_doc_num = i + doc_idx + 1
            doc_ppl = math.exp((d_loss / d_tokens).item())
            print(f"Document {global_doc_num} Perplexity: {doc_ppl:.4f}")
        # -----------------------------------------------------------

        total_loss += masked_loss.sum().item()
        total_tokens += shift_mask.sum().item()

    # compute overall perplexity
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    print(f"Overall Perplexity: {ppl:.4f}")

    return ppl


def main():
    enable_tf32()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--documents",
        type=str,
        required=True,
        help="a jsonl file with a list of strings as documents for perplexity calculation. See data/prefixes.jsonl for an example.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="number of prefixes to batch together during generation",
    )

    args = parser.parse_args()
    with open(args.documents) as f:
        documents = [json.loads(line)["document"] for line in f]
    batch_size = args.batch_size
    device = determine_device()

    # initialize pythia tokenizer and model
    model_name = "EleutherAI/pythia-1.4b"
    tokenizer, model = initialize_pythia(model_name, device)

    # generate and save outputs
    model.eval()
    compute_perplexity(
        model,
        device,
        tokenizer,
        documents,
        batch_size,
    )
    print("done!")


if __name__ == "__main__":
    main()
