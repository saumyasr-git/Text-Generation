import argparse
import json
import os

import torch
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from lm.utils import determine_device, enable_tf32


def initialize_model(model_name: str, device: str):
    """Initialize tokenizer and model for perplexity calculation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return tokenizer, model


@torch.inference_mode()
def calculate_perplexity(model, device, tokenizer, documents: list[str], batch_size: int):
    """Calculate perplexity for a list of documents."""
    total_nll = 0
    total_tokens = 0

    for i in tqdm(range(0, len(documents), batch_size), desc="Calculating Perplexity"):
        batch = documents[i : i + batch_size]
        # Tokenize documents and move to device
        tokenized_batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
        input_ids = tokenized_batch.input_ids
        attention_mask = tokenized_batch.attention_mask

        # Get model outputs
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        # Calculate NLL (Negative Log Likelihood)
        # We need to consider the actual number of non-padded tokens for accurate perplexity
        active_loss = attention_mask.view(-1) == 1
        total_nll += loss.item() * torch.sum(active_loss).item()
        total_tokens += torch.sum(active_loss).item()

    # Perplexity is exp(average negative log likelihood)
    if total_tokens > 0:
        perplexity = torch.exp(torch.tensor(total_nll / total_tokens)).item()
    else:
        perplexity = float('inf')  # Handle case with no tokens

    return perplexity


def main():
    enable_tf32()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--documents",
        type=str,
        required=True,
        help="A jsonl file with a list of documents (strings) to evaluate perplexity on."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Number of documents to batch together during perplexity calculation."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2", # Default to a common model, can be overridden
        help="The name of the model to use for perplexity calculation (e.g., 'gpt2', 'EleutherAI/pythia-1.4b')."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the model on (e.g., 'cuda', 'cpu'). If None, determined automatically."
    )

    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project="text-generation-perplexity", config=args)

    with open(args.documents) as f:
        # CHANGE: Read 'generation' key instead of 'document'
        documents = [json.loads(line)["generation"] for line in f]

    batch_size = args.batch_size
    model_name = args.model
    device = args.device if args.device else determine_device()

    tokenizer, model = initialize_model(model_name, device)

    model.eval()
    perplexity = calculate_perplexity(model, device, tokenizer, documents, batch_size)

    print(f"Perplexity: {perplexity}")

    # Log perplexity to wandb
    wandb.log({"perplexity": perplexity})

    print("done!")
    wandb.finish()


if __name__ == "__main__":
    main()
