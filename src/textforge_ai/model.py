"""Model + tokenizer loading shared by training and inference.

Needs the optional ``train`` dependencies (torch, transformers). Two ways to get a model:

* ``load(name_or_dir)`` - a pretrained checkpoint (default ``Salesforce/codet5-small``,
  downloaded once from Hugging Face) or a directory saved by our trainer;
* ``create_tiny(corpus)`` - a minuscule randomly initialised T5 with a tokenizer trained
  on the fly. It learns almost nothing, but it lets the whole pipeline (train, save,
  load, predict) be tested offline in seconds. Used by the test suite and ``--tiny``.
"""

from __future__ import annotations

from collections.abc import Iterable

DEFAULT_MODEL = "Salesforce/codet5-small"
MAX_SOURCE_TOKENS = 512
MAX_TARGET_TOKENS = 128


def quiet() -> None:
    """Hide per-file progress bars from transformers (they flood the training log)."""
    from transformers.utils import logging

    logging.disable_progress_bar()
    logging.set_verbosity_error()


def pick_device(requested: str = "auto"):
    import torch

    if requested != "auto":
        return torch.device(requested)
    # ROCm builds of PyTorch (AMD GPUs) also report themselves through torch.cuda
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(name_or_dir: str = DEFAULT_MODEL):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    quiet()
    tokenizer = AutoTokenizer.from_pretrained(name_or_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(name_or_dir)
    return tokenizer, model


def create_tiny(corpus: Iterable[str], vocab_size: int = 600):
    from tokenizers import ByteLevelBPETokenizer
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

    quiet()
    special = ["<pad>", "</s>", "<unk>"]
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(list(corpus), vocab_size=vocab_size, special_tokens=special)
    eos_id = bpe.token_to_id("</s>")
    bpe.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", eos_id)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=bpe._tokenizer, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
    )
    config = T5Config(
        vocab_size=len(tokenizer), d_model=64, d_ff=128, d_kv=16, num_heads=4,
        num_layers=2, num_decoder_layers=2, dropout_rate=0.0,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
    )
    return tokenizer, T5ForConditionalGeneration(config)


def encode_batch(tokenizer, sources: list[str], targets: list[str] | None = None, device=None):
    """Tokenize a batch. Padding in the labels becomes -100 (ignored by the loss)."""
    # T5 takes no token_type_ids; some tokenizers emit them by default (transformers 4.x)
    batch = tokenizer(sources, max_length=MAX_SOURCE_TOKENS, truncation=True,
                      padding=True, return_tensors="pt", return_token_type_ids=False)
    if targets is not None:
        labels = tokenizer(text_target=targets, max_length=MAX_TARGET_TOKENS, truncation=True,
                           padding=True, return_tensors="pt")["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        batch["labels"] = labels
    return {k: v.to(device) for k, v in batch.items()} if device is not None else dict(batch)
