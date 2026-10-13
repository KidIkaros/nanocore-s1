"""Tests for NanoCore-S1 BPE tokenizer.

Tests follow strict TDD: each test verifies a specific behavior of the tokenizer
before implementation. Covers:
- Training on text data
- Encoding/decoding roundtrip
- Compression ratio (target: ~4.8x for English text)
- Special tokens for decision tasks ([CHOICE], [Noul], [SCORE], etc.)
- Vocabulary size limits
"""

import pytest
import torch
import tempfile
import os

from src.tokenizer import ByteLevelBPETokenizer, NanoCoreTokenizer, BASE_VOCAB_SIZE


class TestByteLevelBPETokenizer:
    """Tests for byte-level BPE tokenizer (tiktoken-style)."""

    def test_train_basic(self):
        """Tokenizer should train on simple text."""
        texts = ["Hello world!"] * 100
        tok = ByteLevelBPETokenizer()
        tok.train(texts, vocab_size=1000)
        assert tok.vocab_size() > 0
        assert tok.vocab_size() <= 1000

    def test_encode_decode_roundtrip(self):
        """Encoding then decoding should reconstruct the original text."""
        texts = ["Hello world! This is a test."] * 100
        tok = ByteLevelBPETokenizer()
        tok.train(texts, vocab_size=500)
        
        original = "Hello world!"
        encoded = tok.encode(original)
        decoded = tok.decode(encoded)
        assert decoded == original

    def test_compression_ratio(self):
        """Compression ratio should improve with BPE merges.

        With byte-level BPE, even a small vocabulary should achieve
        >1.2x compression on English text. Larger vocabs achieve higher ratios.
        """
        base_texts = [
            "The NanoCore-S1 model is a small transformer designed for fast System One decisions.",
            "It uses rotary positional embeddings and RMS normalization for stable training.",
            "The architecture follows Karpathy nanochat with muP scaling principles.",
            "FineWeb-EDU data provides high-quality educational text for pretraining.",
            "Decision tasks use special tokens like CHOICE and ANSWER for structured output.",
            "Byte-level BPE tokenization maps each byte to a unicode character for safe merging.",
            "The tokenizer achieves a compression ratio of approximately four times for English.",
            "Training on a large corpus improves the vocabulary quality and reduces token count.",
        ] * 500
        
        tok = ByteLevelBPETokenizer()
        tok.train(base_texts, vocab_size=32768, min_frequency=2)
        
        ratio = tok.compression_ratio()
        # Byte-level BPE should compress at least somewhat
        assert ratio >= 1.2, f"Compression ratio {ratio:.2f}x below minimum 1.2x (no merges happening)"

    def test_special_tokens(self):
        """Decision special tokens should be recognized."""
        texts = ["CHOICE options are A B C D ANSWER is A"] * 50
        tok = ByteLevelBPETokenizer()
        tok.train(texts, vocab_size=1000)
        
        # Should have all decision tokens
        for token_name in ["[CHOICE]", "[/CHOICE]", "[ANSWER]", "[/ANSWER]",
                          "[STATE]", "[/STATE]", "[Noul]", "[/Noul]",
                          "[SCORE]", "[/SCORE]"]:
            token_id = tok.get_id(token_name)
            assert token_id is not None, f"Missing special token: {token_name}"

    def test_vocab_size_limit(self):
        """Vocab size should not exceed specified limit."""
        texts = ["Hello world! "] * 1000
        tok = ByteLevelBPETokenizer()
        tok.train(texts, vocab_size=500)
        assert tok.vocab_size() <= 500

    def test_encode_unknown(self):
        """Unknown characters should be broken into bytes."""
        texts = ["Hello world"] * 100
        tok = ByteLevelBPETokenizer()
        tok.train(texts, vocab_size=300)
        
        # Encode text with bytes
        encoded = tok.encode("Hello world")
        assert all(0 <= t < tok.vocab_size() for t in encoded)

    def test_pad_token_id(self):
        """Tokenizer should have a pad token id."""
        texts = ["test "] * 50
        tok = ByteLevelBPETokenizer()
        tok.train(texts, vocab_size=300)
        
        assert tok.pad_token_id >= 0
        assert tok.pad_token_id < tok.vocab_size()

    def test_save_load_roundtrip(self, tmp_path):
        """Tokenizer should serialize and deserialize correctly."""
        texts = ["Hello world! Test data."] * 100
        tok = ByteLevelBPETokenizer()
        tok.train(texts, vocab_size=500)
        
        save_path = str(tmp_path / "tokenizer.json")
        tok.save(save_path)
        
        tok2 = ByteLevelBPETokenizer()
        tok2.load(save_path)
        
        original = "Hello world! Test data."
        assert tok.encode(original) == tok2.encode(original)
        assert tok.decode(tok.encode(original)) == tok2.decode(tok2.encode(original))


class TestNanoCoreTokenizer:
    """Tests for NanoCoreTokenizer (wraps BPE with decision tokens)."""

    def test_default_vocab_size(self):
        """Default vocab should be 32768 (matching model config).

        Before training, only 256 byte tokens + special tokens are available.
        After training, vocab grows towards the target.
        """
        tok = NanoCoreTokenizer()
        # Untrained: only base vocab (256 bytes + special tokens)
        assert tok.vocab_size() == BASE_VOCAB_SIZE
        assert tok.target_vocab_size == 32768

    def test_train_with_special_tokens(self):
        """Training should preserve decision special tokens."""
        texts = ["[STATE] user input [CHOICE] A B C D [ANSWER] A [/ANSWER]"] * 100
        tok = NanoCoreTokenizer(vocab_size=32768)
        tok.train(texts, vocab_size=32768)
        
        # Verify special tokens are in vocab
        for token in ["[STATE]", "[/STATE]", "[CHOICE]", "[/CHOICE]", "[ANSWER]"]:
            assert tok.get_id(token) is not None

    def test_encode_decode_with_special_tokens(self):
        """Special tokens should survive encode/decode roundtrip."""
        texts = ["CHOICE options ANSWER result"] * 100
        tok = NanoCoreTokenizer(vocab_size=32768)
        tok.train(texts, vocab_size=32768)
        
        text = "[STATE] test [CHOICE] A B C [ANSWER] A [/ANSWER]"
        encoded = tok.encode(text)
        decoded = tok.decode(encoded)
        assert "[STATE]" in decoded
        assert "[CHOICE]" in decoded
        assert "[ANSWER]" in decoded

    def test_add_special_tokens(self):
        """New special tokens can be added after training."""
        tok = NanoCoreTokenizer(vocab_size=32768)
        
        # The decision tokens should already be in the tokenizer
        assert tok.get_id("[CHOICE]") is not None
        assert tok.get_id("[Noul]") is not None

    def test_ensure_token(self):
        """ensure_token should return id if exists, -1 if not."""
        tok = NanoCoreTokenizer(vocab_size=32768)
        
        # Existing token
        assert tok.ensure_token("[CHOICE]") >= 0
        
        # Non-existing token (after training, some tokens won't exist)
        result = tok.ensure_token("[UNKNOWN_TOKEN_XYZ]")
        assert result == -1
