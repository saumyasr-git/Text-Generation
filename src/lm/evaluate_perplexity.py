@torch.inference_mode()
def compute_perplexity(
    model: AutoModelForCausalLM,
    device: str,
    tokenizer: AutoTokenizer,
    documents: list[str],
    batch_size: int,
) -> list[float]:
    """Computes perplexity per document given a list of documents.

    Args:
        model: the language model
        device: device to put the tensors on
        tokenizer: the tokenizer
        documents: a list of document strings
        batch_size: number of documents to batch together during evaluation

    Returns:
        perplexities: a list of floating point perplexity values for each document
    """
    perplexities = []

    # Process batch by batch
    for i in trange(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        tokenized_docs = tokenizer(batch_docs, return_tensors="pt", padding=True)

        iids = tokenized_docs["input_ids"].to(device)
        mask = tokenized_docs["attention_mask"].to(device)

        logits = model(input_ids=iids, attention_mask=mask).logits

        # Shift sequence for standard autoregressive language modeling targets
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = iids[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous()

        # Compute unreduced cross-entropy loss per token
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.size())

        # Compute loss and perplexity per document row in the batch
        doc_losses = (loss * shift_mask).sum(dim=1)
        doc_tokens = shift_mask.sum(dim=1)

        for doc_idx, (d_loss, d_tokens) in enumerate(zip(doc_losses, doc_tokens)):
            global_doc_num = i + doc_idx + 1
            avg_doc_loss = (d_loss / d_tokens).item()
            doc_ppl = math.exp(avg_doc_loss)
            perplexities.append(doc_ppl)
            print(f"Document {global_doc_num} Perplexity: {doc_ppl:.4f}")

    return perplexities
