"""
NanoCore-S1 Byte-Level BPE Tokenizer

Implements a byte-level BPE tokenizer with decision-task special tokens,
following tiktoken's approach (Karpathy's nanochat uses a similar tokenizer).

Architecture:
1. Start with 256 byte tokens (one per byte value)
2. Add special tokens for decision tasks
3. Train BPE merges on text data
4. Vocabulary size capped at specified limit

Special tokens (Jev decision primitives):
- [STATE]     / [/STATE]    — Context/state marker
- [CHOICE]    / [/CHOICE]   — Choice options list
- [ANSWER]    / [/ANSWER]   — Selected answer
- [Noul]      / [/Noul]     — Negative/null option (type safety)
- [SCORE]     / [/SCORE]    — Numerical confidence score

Based on tiktoken's byte-pair encoding approach.
Reference: https://github.com/openai/tiktoken (BSD-3-Clause)
"""

import re
import json
import os
from collections import Counter
from typing import List, Dict, Optional, Tuple
from pathlib import Path


# Decision-task special tokens (Jev System One primitives)
DECISION_SPECIAL_TOKENS = [
    "[STATE]", "[/STATE]",
    "[CHOICE]", "[/CHOICE]",
    "[ANSWER]", "[/ANSWER]",
    "[Noul]", "[/Noul]",
    "[SCORE]", "[/SCORE]",
    "[ROUTE]", "[/ROUTE]",
    "[CONFIDENCE]", "[/CONFIDENCE]",
]

# Additional special tokens
ADDITIONAL_SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|sep|>",
    "<|mask|>",
]

# All special tokens (reserved at vocab start)
ALL_SPECIAL_TOKENS = ADDITIONAL_SPECIAL_TOKENS + DECISION_SPECIAL_TOKENS
NUM_SPECIAL_TOKENS = len(ALL_SPECIAL_TOKENS)  # 20 special tokens

# Start vocab with 256 bytes + special tokens
BASE_VOCAB_SIZE = 256 + NUM_SPECIAL_TOKENS


def bytes_to_unicode():
    """Return a dict mapping bytes to unicode characters.

    This prevents special regex characters from being part of the BPE
    by mapping bytes to a clean unicode space.
    Uses the same approach as tiktoken/GPT-2.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("©"), ord("ë") + 1)) + \
         list(range(ord("ì"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(0xe000 + n)  # Use private use area
            n += 1
    bs = [b for b in bs]  # bs contains only byte values 0-255
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs)), dict(zip(cs, bs))


def get_byte_pairs(word: str) -> set:
    """Get all consecutive pairs in a word."""
    pairs = set()
    if len(word) < 2:
        return pairs
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class ByteLevelBPETokenizer:
    """Byte-level BPE tokenizer (tiktoken-style simplicity).

    Simplified implementation suitable for small-scale training:
    - Byte-level encoding (256 base tokens)
    - BPE merge algorithm
    - Special token support
    - JSON serialization
    """

    def __init__(self):
        # Byte <-> unicode mappings
        self.b2u, self.u2b = bytes_to_unicode()
        
        # Token -> id and id -> token mappings
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # BPE merge rules (list of token pairs to merge)
        self.merges: List[Tuple[str, str]] = []
        
        # Special token IDs
        self.special_tokens = ALL_SPECIAL_TOKENS
        
        # Initialize with base vocabulary
        self._init_base_vocab()
        
        # Track whether training has been done
        self._trained = False

    def _init_base_vocab(self):
        """Initialize vocabulary with bytes and special tokens."""
        # First 256 entries are bytes (0-255)
        for i in range(256):
            token = self.b2u[i]
            self.id_to_token[i] = token
            self.token_to_id[token] = i
        
        # Next entries are special tokens
        for i, token in enumerate(self.special_tokens):
            token_id = 256 + i
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

    def _encode_text_to_unicode(self, text: str) -> str:
        """Convert text to unicode representation for BPE."""
        # Handle special tokens first
        for token in self.special_tokens:
            if token in text:
                # Temporarily replace with a placeholder
                pass
        
        # Convert bytes to unicode
        encoded = text.encode("utf-8")
        return "".join(self.b2u[b] for b in encoded)

    def _pre_tokenize(self, text: str) -> List[str]:
        """Pre-tokenize text into word-level chunks.

        Uses a regex-based approach similar to GPT-2/3.
        """
        # Pattern: words, numbers, punctuation, special tokens
        pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|
                      (?:\[[A-Z]+\])|(?:<\|[^>]+\|>)|
                      \s+"""
        
        # First, extract all special tokens
        special_pattern = r"(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
        parts = re.split(f"({special_pattern[1:-1]})", text)
        
        words = []
        for part in parts:
            if part in self.special_tokens:
                words.append(part)
            elif part.strip():
                # Standard regex tokenization
                words.extend(re.findall(r"\S+|\s+", part))
        
        # Convert each word to unicode representation
        result = []
        for word in words:
            if word in self.special_tokens:
                result.append(word)
            else:
                result.append(self._encode_text_to_unicode(word))
        
        return result

    def _bpe_merge(self, word: str, merges: List[Tuple[str, str]]) -> List[str]:
        """Apply BPE merges to a single word, returning subwords.

        Args:
            word: Word in unicode representation
            merges: List of (first, second) pairs to merge

        Returns:
            List of subword tokens
        """
        if word in self.special_tokens:
            return [word]
        
        # Convert word to individual characters
        chars = list(word)
        
        # Apply merges in order
        for first, second in merges:
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == first and chars[i+1] == second:
                    new_chars.append(first + second)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
        
        return chars if chars else [""]

    def train(self, texts: List[str], vocab_size: int = 32768,
              min_frequency: int = 2, min_remaining_bytes: int = 1):
        """Train BPE merges on the given texts.

        Args:
            texts: List of training texts
            vocab_size: Target vocabulary size (including base 256 bytes + special tokens)
            min_frequency: Minimum frequency for a merge to be considered
            min_remaining_bytes: Minimum bytes to keep in vocab (from tiktoken)
        """
        target_merges = vocab_size - BASE_VOCAB_SIZE
        
        # Count all word occurrences
        word_counts = Counter()
        for text in texts:
            words = self._pre_tokenize(text)
            for word in words:
                word_counts[word] += 1
        
        print(f"  Pre-tokenized: {len(word_counts)} unique words, {sum(word_counts.values())} total")
        
        # Get character-level representation for each word
        # word_freqs maps word -> (count, list of individual chars)
        word_freqs = {}
        for word, count in word_counts.items():
            if word in self.special_tokens:
                word_freqs[word] = (count, [])
            else:
                word_freqs[word] = (count, list(word))
        
        # BPE merge loop
        vocab_size_current = BASE_VOCAB_SIZE
        merge_count = 0
        # Track the next available token id
        token_id_counter = BASE_VOCAB_SIZE
        
        while vocab_size_current < vocab_size and merge_count < target_merges:
            # Count all adjacent pairs
            pairs = Counter()
            for word, (count, chars) in word_freqs.items():
                if word in self.special_tokens:
                    continue
                if len(chars) < 2:
                    continue
                for i in range(len(chars) - 1):
                    pairs[(chars[i], chars[i+1])] += count
            
            # Filter by minimum frequency
            pairs = {pair: freq for pair, freq in pairs.items() if freq >= min_frequency}
            
            if not pairs:
                print(f"  No more pairs to merge (all below min_frequency={min_frequency})")
                break
            
            # Find the most frequent pair
            best_pair = max(pairs, key=lambda p: pairs[p])
            
            # Add merge
            self.merges.append(best_pair)
            vocab_size_current += 1
            merge_count += 1
            
            # Apply merge to all words
            first, second = best_pair
            new_token = first + second
            
            # Add new token to vocab
            new_id = token_id_counter
            token_id_counter += 1
            self.id_to_token[new_id] = new_token
            self.token_to_id[new_token] = new_id
            
            # Update word_freqs
            for word in list(word_freqs.keys()):
                count, chars = word_freqs[word]
                if word in self.special_tokens:
                    continue
                
                new_chars = []
                i = 0
                while i < len(chars):
                    if i < len(chars) - 1 and chars[i] == first and chars[i+1] == second:
                        new_chars.append(new_token)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                word_freqs[word] = (count, new_chars)
            
            if merge_count % 5000 == 0:
                print(f"  Merges: {merge_count}/{target_merges} (vocab {vocab_size_current})")
        
        self._trained = True
        print(f"  Training complete: {merge_count} merges, vocab size {self.vocab_size()}")

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids.

        Args:
            text: Input text (may contain special tokens)

        Returns:
            List of token ids
        """
        words = self._pre_tokenize(text)
        token_ids = []
        
        for word in words:
            if word in self.special_tokens:
                token_id = self.token_to_id.get(word)
                if token_id is not None:
                    token_ids.append(token_id)
                continue
            
            # Apply BPE merges
            subwords = self._bpe_merge(word, self.merges)
            for sw in subwords:
                if sw == "":
                    continue
                token_id = self.token_to_id.get(sw)
                if token_id is not None:
                    token_ids.append(token_id)
                else:
                    # Byte-level fallback: encode character by character
                    for ch in sw:
                        byte_val = self.u2b.get(ch)
                        if byte_val is not None and byte_val < 256:
                            token_ids.append(byte_val)
                        else:
                            # Unknown character — skip
                            pass
        
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids back to text.

        Args:
            token_ids: List of token ids

        Returns:
            Decoded text
        """
        parts = []
        for token_id in token_ids:
            if token_id not in self.id_to_token:
                continue  # Skip unknown token ids
            
            token = self.id_to_token[token_id]
            if token in self.special_tokens:
                parts.append(token)
            else:
                # Token is a unicode string; map each char back to byte
                byte_values = []
                for ch in token:
                    byte_val = self.u2b.get(ch)
                    if byte_val is not None:
                        byte_values.append(byte_val)
                parts.append(bytes(byte_values))
        
        # Join all parts and decode to UTF-8
        all_bytes = b"".join(p if isinstance(p, bytes) else p.encode("utf-8") for p in parts)
        return all_bytes.decode("utf-8", errors="replace")

    def vocab_size(self) -> int:
        """Return current vocabulary size."""
        return len(self.id_to_token)

    def get_id(self, token: str) -> Optional[int]:
        """Look up token id by token string."""
        return self.token_to_id.get(token)

    def get_token(self, token_id: int) -> Optional[str]:
        """Look up token string by id."""
        return self.id_to_token.get(token_id)

    @property
    def pad_token_id(self) -> int:
        """ID for padding token."""
        return self.token_to_id.get("<|pad|>", 256)

    @property
    def bos_token_id(self) -> int:
        """ID for beginning-of-sequence token."""
        return self.token_to_id.get("<|bos|>", 257)

    @property
    def eos_token_id(self) -> int:
        """ID for end-of-sequence token."""
        return self.token_to_id.get("<|eos|>", 258)

    def compression_ratio(self) -> float:
        """Calculate compression ratio on a sample of English text.

        Compression ratio = (total characters) / (total tokens)
        For English text with byte-level BPE, target is ~4.0-6.0x
        """
        sample_text = (
            "The quick brown fox jumps over the lazy dog. " * 100 +
            "NanoCore-S1 is a small transformer model. " * 50 +
            "System One decisions require fast and accurate classification. " * 50
        )
        
        char_count = len(sample_text)
        token_ids = self.encode(sample_text)
        token_count = len(token_ids)
        
        if token_count == 0:
            return 1.0
        
        return char_count / token_count

    def save(self, path: str):
        """Save tokenizer to JSON file."""
        data = {
            "merges": [[first, second] for first, second in self.merges],
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "token_to_id": self.token_to_id,
            "special_tokens": self.special_tokens,
        }
        
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Tokenizer saved to {path}")

    def load(self, path: str):
        """Load tokenizer from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        self.merges = [tuple(m) for m in data["merges"]]
        self.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        self.token_to_id = data["token_to_id"]
        self.special_tokens = data.get("special_tokens", ALL_SPECIAL_TOKENS)
        self._trained = True
        print(f"Tokenizer loaded from {path} (vocab_size={self.vocab_size()})")


class NanoCoreTokenizer(ByteLevelBPETokenizer):
    """NanoCore-S1 tokenizer with decision-task special tokens.

    Extends ByteLevelBPETokenizer with:
    - Decision task tokens ([STATE], [CHOICE], [ANSWER], etc.)
    - 32768 vocab size by default
    - Easy train/save/load API
    """

    def __init__(self, vocab_size: int = 32768):
        super().__init__()
        self.target_vocab_size = vocab_size

    def train(self, texts: List[str], vocab_size: int = 32768,
              min_frequency: int = 2, min_remaining_bytes: int = 1):
        """Train tokenizer on text data.

        Special tokens are reserved first, then BPE merges are learned
        from the text data.
        """
        super().train(texts, vocab_size=vocab_size,
                     min_frequency=min_frequency,
                     min_remaining_bytes=min_remaining_bytes)

    def ensure_token(self, token: str) -> int:
        """Look up a token id, return -1 if not found.

        Useful for checking if decision tokens exist in the vocabulary.
        """
        return self.token_to_id.get(token, -1)


if __name__ == "__main__":
    # Quick test
    texts = [
        "Hello world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "[STATE] user query [CHOICE] option A option B [ANSWER] option A [/ANSWER]",
    ] * 100
    
    tok = NanoCoreTokenizer(vocab_size=32768)
    tok.train(texts, vocab_size=32768)
    
    print(f"Vocab size: {tok.vocab_size()}")
    print(f"Compression ratio: {tok.compression_ratio():.2f}x")
    print(f"Special tokens: {[(t, tok.token_to_id(t)) for t in ['[STATE]', '[CHOICE]', '[ANSWER]', '[Noul]']]}")
