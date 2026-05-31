from collections import defaultdict


class BPETokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []  # ordered list of merge rules
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def train(self, corpus: list[str]):
        # Build initial word frequencies: "hello" -> {"h e l l o </w>": 3}
        word_freq: dict[str, int] = defaultdict(int)
        for text in corpus:
            for word in text.strip().split():
                tokenized = " ".join(list(word)) + " </w>"
                word_freq[tokenized] += 1

        # Seed vocab with all unique characters + </w>
        vocab: set[str] = set()
        for word in word_freq:
            vocab.update(word.split())

        while len(vocab) < self.vocab_size:
            pair_freq = self._count_pairs(word_freq)
            if not pair_freq:
                break
            best = max(pair_freq, key=pair_freq.get)
            word_freq = self._merge_pair(best, word_freq)
            new_token = best[0] + best[1]
            vocab.add(new_token)
            self.merges.append(best)

        # Build final token <-> id mappings
        for idx, token in enumerate(sorted(vocab)):
            self.vocab[token] = idx
            self.id_to_token[idx] = token

    def _count_pairs(self, word_freq: dict) -> dict:
        pairs: dict[tuple, int] = defaultdict(int)
        for word, freq in word_freq.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_pair(self, pair: tuple[str, str], word_freq: dict) -> dict:
        merged = pair[0] + pair[1]
        new_word_freq: dict[str, int] = {}
        bigram = " ".join(pair)
        for word, freq in word_freq.items():
            new_word = word.replace(bigram, merged)
            new_word_freq[new_word] = freq
        return new_word_freq

    # ------------------------------------------------------------------ #
    #  Encoding / Decoding                                                 #
    # ------------------------------------------------------------------ #

    def _tokenize_word(self, word: str) -> list[str]:
        symbols = list(word) + ["</w>"]
        for left, right in self.merges:
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == left and symbols[i + 1] == right:
                    symbols = symbols[:i] + [left + right] + symbols[i + 2:]
                else:
                    i += 1
        return symbols

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in text.strip().split():
            for token in self._tokenize_word(word):
                if token in self.vocab:
                    ids.append(self.vocab[token])
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.id_to_token.get(i, "") for i in ids]
        text = "".join(tokens)
        return text.replace("</w>", " ").strip()


if __name__ == "__main__":
    corpus = [
        "low low low low low",
        "lower lower",
        "newest newest newest newest",
        "widest widest",
    ]

    tokenizer = BPETokenizer(vocab_size=30)
    tokenizer.train(corpus)

    print("Merges learned:")
    for merge in tokenizer.merges:
        print(" ", merge)

    print("\nVocab:", sorted(tokenizer.vocab.keys()))

    test = "lowest newest"
    ids = tokenizer.encode(test)
    decoded = tokenizer.decode(ids)
    print(f"\nInput:   '{test}'")
    print(f"Token ids: {ids}")
    print(f"Decoded: '{decoded}'")
