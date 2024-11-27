import re
from collections import defaultdict
from typing import List

from pretty_midi import PrettyMIDI


class MIDI_Tokenizer:
    def __init__(self, encoder, decoder, vocab):
        self.merges = {}  # (int, int) -> int
        self.encoder = encoder
        self.decoder = decoder
        self.tokens_list = []
        self.merged_tokens_list = []
        self.vocab = vocab

    def build_vocab(self):
        raise NotImplementedError

    def encode_all(self, midi_files):
        for genre in midi_files:
            for file in midi_files[genre]:
                self.tokens_list.append(self.encode(file, genre))

    def encode(self, score: "PrettyMIDI", genre, readable=False):
        return self.encoder.encode(score, genre, readable)

    def decode(self, score: "PrettyMIDI", readable=False):
        return self.decoder.decode(score, readable)

    def train(self, midi_files, vocab_size, readable=False):
        """
        A method to the train the tokenizer on a dataset of pretty midi files, to learn
        the most common merges it needs to perform.
        """
        self.encode_all(midi_files)
        split_tokens = [self.split_tokens(tokens) for tokens in self.tokens_list]
        merged_tokens_list = list(split_tokens)
        num_merges = vocab_size - len(self.vocab)
        new_token = len(self.vocab)
        for _ in range(num_merges):
            pair_counts = self.count_all_pairs(merged_tokens_list, readable)
            max_pair = max(pair_counts, key=pair_counts.get)
            if max_pair[0] < len(self.vocab) and max_pair[1] < len(self.vocab):
                token_1, token_2 = (
                    self.vocab.inv[max_pair[0]],
                    self.vocab.inv[max_pair[1]],
                )
            else:
                token_1, token_2 = max_pair
            merged_tokens_list = self.merge(merged_tokens_list, max_pair, new_token)
            self.merges[max_pair] = new_token
            new_token += 1

        self.merged_tokens_list = merged_tokens_list

    def split_tokens(self, tokens, readable=False):
        """
        Splits a list of tokens into chunks grouped by matching patterns.

        Parameters:
            tokens (list): A list of token strings to process.

        Returns:
            list: A new list where tokens are grouped into chunks based on patterns.
        """

        def get_token_type(token):
            if re.match(r"^(Vel|Pitch|Octave|Duration)_", token):
                return "VPOD"
            if re.match(r"^(Bar|Beat|Pos)", token):
                return "POS"
            return "Separate"

        new_tokens = []  # Resulting list of tokens or grouped chunks
        tokens_chunk = []  # Current chunk of tokens being grouped
        token_type = last_token_type = None

        for token in tokens:
            readable_token = self.get_readable_token(token, readable)
            token_type = get_token_type(readable_token)
            # If the token group changes
            if (
                tokens_chunk
                and token_type != last_token_type
                or (token_type == "VPOD"
                and last_token_type == "VPOD"
                and len(tokens_chunk) >= 4)
            ):
                new_tokens.append(tokens_chunk)
                tokens_chunk = []

                if token_type == "Separate":
                    new_tokens.append(token)
                else:
                    tokens_chunk = [token]

            else:
                if token_type == "Separate":
                    new_tokens.append(token)
                else:
                    tokens_chunk.append(token)

            last_token_type = token_type

        if tokens_chunk:
            new_tokens.append(tokens_chunk)

        return new_tokens

    def merge(self, tokens_list, pair, new_token):
        merged_tokens_list = []
        for tokens in tokens_list:
            new_tokens = []
            chunk_idx = 0
            while chunk_idx < len(tokens):

                token_chunk = tokens[chunk_idx]
                new_tokens_chunk = []
                i = 0
                while isinstance(token_chunk, list) and i < len(token_chunk):
                    # Check if the current and next token form the pair
                    if (
                        i < len(token_chunk) - 1
                        and (token_chunk[i], token_chunk[i + 1]) == pair
                    ):
                        new_tokens_chunk.append(new_token)  # Merge the pair
                        i += 2  # Skip the pair
                    else:
                        new_tokens_chunk.append(token_chunk[i])  # Add the current token
                        i += 1  # Move to the next token

                if len(new_tokens_chunk) == 1:
                    new_tokens_chunk = new_tokens_chunk[0]

                if new_tokens_chunk:
                    new_tokens.append(new_tokens_chunk)
                    chunk_idx += 1
                elif (
                    chunk_idx < len(tokens) - 1
                    and not isinstance(tokens[chunk_idx + 1], list)
                    and (token_chunk, tokens[chunk_idx + 1]) == pair
                ):
                    new_tokens.append(new_token)  # Merge the pair
                    chunk_idx += 2
                else:
                    new_tokens.append(token_chunk)
                    chunk_idx += 1

            merged_tokens_list.append(new_tokens)

        return merged_tokens_list

    def count_all_pairs(self, tokens_list, readable=False):
        pairs = defaultdict(int)  # Initialize a dictionary to store pair counts
        for tokens in tokens_list:
            idx = 0
            for group in tokens:
                if idx == len(tokens) - 1:
                    break

                if isinstance(group, list):
                    pairs = self.count_pairs(group, pairs)
                elif (
                    self.is_mergeable(group, readable)
                    and self.is_mergeable_forward(group, readable)
                    and (
                        not isinstance(tokens[idx + 1], list)
                        and self.is_mergeable(tokens[idx + 1], readable)
                    )
                ):
                    pairs[group, tokens[idx + 1]] += 1

                idx += 1
        return pairs

    def count_pairs(self, lst, pairs: defaultdict):
        for pair in zip(lst, lst[1:]):
            pairs[pair] += 1
        return pairs

    def get_readable_token(self, token, readable=False):
        if readable:
            return token
        return self.vocab.inv.get(token, "Merged Token")

    def is_mergeable(self, token, readable):
        readable_token = self.get_readable_token(token, readable)
        return not (
            readable_token in ["SOS", "EOS", "EOC"]
            or readable_token.startswith("Time Signature")
            or readable_token.startswith("Key")
            or readable_token.startswith("Tempo")
        )

    def is_mergeable_forward(self, token, readable):
        readable_token = self.get_readable_token(token, readable)
        return readable_token != "EOC"
