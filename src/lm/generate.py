import argparse
import json
import os
import math

import tiktoken
import torch
from omegaconf import OmegaConf
from tqdm import trange
from lm.model import DecoderLM
from lm.utils import determine_device, enable_tf32
from lm.train import compute_language_modeling_loss


def softmax_with_temperature(
    logits: torch.FloatTensor, temperature: float
) -> torch.FloatTensor:
    """Turns logits into probabilities under softmax (with temperature)

    Args:
        logits: a 2d torch tensor of token ids (B x V)
        temperature: temperature of the softmax function

    Returns:
        a 2d torch tensor of token probabilities (B x V)
    """

    # to avoid division by 0
    temperature = max(temperature, 1e-5)
    
    return torch.softmax(logits / temperature, dim=-1)


@torch.inference_mode()
def generate(
    model: DecoderLM,
    device: str,
    tokenizer: tiktoken.Encoding,
    prefixes: list[str],
    batch_size: int,
    max_new_tokens: int = 32,
    temperature: float = 0.1,
) -> list[str]:
    """Generates completions conditioned on prefixes; computes perplexity

    Args:
        model: the language model
        device: device to put the tensors on
        tokenizer: the tokenizer
        prefixes: a list of strings as prefixes for generation
        batch_size: number of prefixes to batch together during generation
        max_new_tokens: the number of tokens to generate for each prefix
        temperature: temperature parameter of softmax

    Returns:
        a list of strings (continuations to prefixes)

    Note: you should implement a batched version of this function by
        left-padding tokenized prefixes with `tokenizer.eot_token` so that all
        sequences have equal length. `attention_mask` should be set to 0.0 for
        padding tokens, and 1.0 everywhere else.
    """
    token_ids = [tokenizer.encode(prefix) for prefix in prefixes]
    longest_prefix_len = max(len(ids) for ids in token_ids)
    padded_token_ids = [
    [tokenizer.eot_token] * (longest_prefix_len - len(ids)) + ids
    for ids in token_ids
]
    losses=[]
    for start in range(0, len(padded_token_ids), batch_size):
        end = start + batch_size


        batch_token_ids = padded_token_ids[start:end]
        original_batch_token_ids = token_ids[start:end]
        batch_token_ids_tensor = torch.tensor(batch_token_ids, device=device)
        attention_mask = torch.tensor([
                    [0.0] * (longest_prefix_len - len(ids)) + [1.0] * len(ids)
                    for ids in original_batch_token_ids], device=device)
        batch_prefixes = prefixes[start:end]
        for _ in trange(max_new_tokens):
            logits = model(batch_token_ids_tensor, attention_mask)  
            next_token_logits = logits[:, -1, :]
            probabilities = softmax_with_temperature(next_token_logits, temperature)
            next_token_ids = torch.multinomial(probabilities, num_samples=1)
            batch_token_ids_tensor = torch.cat([batch_token_ids_tensor, next_token_ids], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), device=device)], dim=1)
        generated_token_ids = batch_token_ids_tensor[:, longest_prefix_len:].tolist()
        loss = compute_language_modeling_loss(token_ids, logits)
        losses.append(loss.item())

    # Process this batch through the model.
    generations = [tokenizer.decode(ids) for ids in generated_token_ids]

    

    # mean of the losses is the average negative log likelihood
    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)
    

    print(f"Perplexity: {perplexity}")
    return generations


def main():
    enable_tf32()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=OmegaConf.load,
        required=True,
        help="the yaml config file used for model training",
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        required=True,
        help="a json file with a list of strings as prefixes for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="temperature in sampling"
    )

    args = parser.parse_args()
    config = args.config
    with open(args.prefixes) as f:
        prefixes = [json.loads(line)["prefix"] for line in f]
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature

    # initialize tokenizer and model
    model_path = os.path.join(config.output_dir, "model.pt")
    assert os.path.exists(model_path), f"no model checkpoint at {model_path}"
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    device = determine_device() if config.device == "auto" else config.device
    model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # generate and save outputs
    model.eval()
    generations = generate(
        model,
        device,
        tokenizer,
        prefixes,
        config.batch_size,
        max_new_tokens,
        temperature,
    )

    generation_path = os.path.join(config.output_dir, "generation.jsonl")
    print(f"writing generations to {generation_path}")
    with open(generation_path, "w") as f:
        for prefix, generation in zip(prefixes, generations):
            json.dump({"prefix": prefix, "generation": generation}, f)
            f.write("\n")

    print("done!")


if __name__ == "__main__":
    main()