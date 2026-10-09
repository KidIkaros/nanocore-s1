#!/usr/bin/env python
"""Tokenizer training script for NanoCore-S1.

Trains a byte-level BPE tokenizer on FineWeb-EDU data and saves to disk.

Usage:
    python scripts/train_tokenizer.py [--vocab-size=32768] [--output=tokenizer.json]
    python scripts/train_tokenizer.py --dry-run  # Uses small sample, quick test

The tokenizer includes decision-task special tokens required by Jev:
    [STATE], [CHOICE], [ANSWER], [Noul], [SCORE], etc.

References:
    - tiktoken (OpenAI): https://github.com/openai/tiktoken
    - Karpathy's nanochat: similar byte-level BPE approach
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenizer import NanoCoreTokenizer, BASE_VOCAB_SIZE


def load_finetune_edu_sample(max_chars: int = None):
    """Load FineWeb-EDU dataset sample.

    Uses the HuggingFace datasets library to stream FineWeb-EDU.
    Falls back to local samples if datasets not available.

    Args:
        max_chars: Maximum characters to read (for testing/quotas)

    Returns:
        List of text strings
    """
    print("Loading FineWeb-EDU data...")

    try:
        from datasets import load_dataset
        # Stream to avoid downloading entire 300GB dataset
        ds = load_dataset("mlfoundations/fineweb-edu-10bt", split="train", streaming=True)
        texts = []
        total_chars = 0
        for item in ds:
            text = item.get("text", "")
            if text.strip():
                texts.append(text)
                total_chars += len(text)
                if max_chars and total_chars >= max_chars:
                    break
        print(f"  Loaded {len(texts)} documents, {total_chars} chars")
        return texts
    except ImportError:
        print("  datasets library not available, using fallback text samples")
        return get_sample_texts()


def get_sample_texts():
    """Return sample English text for tokenizer training/debugging.

    These texts cover diverse vocabulary to produce meaningful merges.
    """
    return [
        "The NanoCore-S1 model is a small transformer designed for fast System One decisions.",
        "It uses rotary positional embeddings and RMS normalization for stable training.",
        "The architecture follows Karpathy nanochat with muP scaling principles.",
        "FineWeb-EDU data provides high-quality educational text for pretraining.",
        "Decision tasks use special tokens like CHOICE and ANSWER for structured output.",
        "Byte-level BPE tokenization maps each byte to a unicode character for safe merging.",
        "The tokenizer achieves a compression ratio of approximately four times for English.",
        "Training on a large corpus improves the vocabulary quality and reduces token count.",
        "Subword tokenization helps handle rare words and out-of-vocabulary tokens gracefully.",
        "The muP scaling method ensures stable training across different model sizes.",
        "System One refers to fast, intuitive, automatic cognitive processes.",
        "System Two refers to slow, deliberate, analytical reasoning processes.",
        "Type safety in AI systems prevents certain classes of runtime errors.",
        "The transformer architecture uses self-attention for sequence modeling.",
        "Rotary positional embeddings encode position information in the attention matrix.",
        "Layer normalization stabilizes hidden state distributions across training.",
        "Dropout regularization prevents overfitting in neural network training.",
        "Gradient clipping stabilizes training by limiting gradient magnitude.",
        "Cross-entropy loss is the standard training objective for language models.",
        "AdamW optimizer combines momentum with weight decay for better convergence.",
        "Learning rate schedules control optimization step size over training.",
        "The softmax function normalizes attention weights to sum to one.",
        "Feed-forward networks add non-linearity after each attention layer.",
        "The hidden dimension determines the internal representation size.",
        "Number of attention heads controls how many patterns are learned in parallel.",
    ] * 2000  # Repeat to provide sufficient training data


def train(vocab_size: int, output_path: str, max_chars: int = None):
    """Train tokenizer and save to disk.

    Args:
        vocab_size: Target vocabulary size
        output_path: Path to save tokenizer JSON
    """
    # Initialize tokenizer
    tok = NanoCoreTokenizer(vocab_size=vocab_size)

    # Load training data
    if max_chars is not None:
        print(f"Loading sample data (max {max_chars} chars)...")
        # Use sample texts for dry-run
        texts = get_sample_texts()
    else:
        texts = load_finetune_edu_sample(max_chars=max_chars)

    # Train
    print(f"\nTraining tokenizer with vocab_size={vocab_size}...")
    print(f"  Training on {len(texts)} documents")
    tok.train(texts, vocab_size=vocab_size, min_frequency=2)

    # Report
    print(f"\nTraining complete!")
    print(f"  Vocabulary size: {tok.vocab_size()}")
    print(f"  BPE merges: {len(tok.merges)}")
    print(f"  Compression ratio: {tok.compression_ratio():.2f}x")
    print(f"  Target vocab size: {vocab_size}")

    # Verify special tokens
    print("\n  Special tokens:")
    for token in ["[STATE]", "[CHOICE]", "[ANSWER]", "[Noul]", "[SCORE]"]:
        token_id = tok.ensure_token(token)
        print(f"    {token}: id={token_id}")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tok.save(output_path)
    print(f"\n  Saved to: {output_path}")

    # Also save metadata
    meta_path = output_path.replace(".json", "_meta.json")
    meta = {
        "vocab_size": tok.vocab_size(),
        "target_vocab_size": vocab_size,
        "num_merges": len(tok.merges),
        "compression_ratio": round(tok.compression_ratio(), 4),
        "special_tokens": tok.special_tokens,
        "base_vocab_size": BASE_VOCAB_SIZE,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to: {meta_path}")

    return tok


def main():
    parser = argparse.ArgumentParser(description="Train NanoCore-S1 tokenizer")
    parser.add_argument(
        "--vocab-size", type=int, default=32768,
        help="Target vocabulary size (default: 32768)"
    )
    parser.add_argument(
        "--output", type=str, default="models/tokenizer.json",
        help="Output path for tokenizer JSON (default: models/tokenizer.json)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Quick dry-run: small vocab, sample data only"
    )
    args = parser.parse_args()

    if args.dry_run:
        # Quick test with small vocab and sample data
        print("=== Dry Run Mode ===")
        train(
            vocab_size=min(args.vocab_size, 8192),
            output_path=args.output,
            max_chars=100000,  # ~100K chars for quick test
        )
    else:
        train(
            vocab_size=args.vocab_size,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
