from dataclasses import dataclass

from pretty_midi import PrettyMIDI

from data_structures.linked_array import LinkedArray
from data_structures.max_priority_map import MaxPriorityMap
from data_structures.merge_trie import MergeTrie

@dataclass
class PairHeapItem:
    pair: tuple
    positions: set

class MIDI_Tokenizer:
    def __init__(self, encoder, decoder, vocab):
        self.merges = {}  # (int, int) -> int
        self._encoder = encoder
        self._decoder = decoder
        self._vocab = vocab
        self._merge_trie = MergeTrie(vocab)
        self._encoded_tokens = []
        self._is_trained = False

    def build_vocab(self):
        raise NotImplementedError

    def encode_with_bpe(self, midi_files):
        if not self._is_trained:
            raise ValueError("Tokenizer isn't trained, run train_tokens method first")

        for genre in midi_files:
            for file in midi_files[genre]:
                self._encoded_tokens.append(
                    self._merge_trie._merge(self.encode(file, genre))
                )

    def bpe_test(self, tokens_list):
        return [self._merge_trie._merge(tokens) for tokens in tokens_list]

    def get_encoded_tokens(self):
        return self._encoded_tokens

    def encode(self, score: "PrettyMIDI", genre="Unknown", readable=False):
        return self._encoder.encode(score, genre, readable)

    def decode(self, score: "PrettyMIDI", readable=False):
        return self._decoder.decode(score, readable)

    def train_tokens(
        self,
        tokens_list,
        vocab_size,
        get_token_type=None,
        is_unmergeable=None,
        is_mergeable_token_types=None,
    ):

        self._merge_trie = MergeTrie(self._vocab)
        self._encoded_tokens = []

        if is_unmergeable:
            self._is_unmergeable = is_unmergeable

        if is_mergeable_token_types:
            self.is_mergeable_token_types = is_mergeable_token_types

        if get_token_type:
            self._get_token_type = get_token_type

        self._tokens_list = [LinkedArray(tokens) for tokens in tokens_list]
        self._pair_heap = self._create_pair_heap()
        num_merges = vocab_size - len(self._vocab)

        print("Beginning Merges...")
        new_token = len(self._vocab)
        for _ in range(num_merges):
            max_pair = self._pair_heap.pop()
            self._merge_pair_and_update_heap(max_pair, new_token)
            self._merge_trie._insert_merge(max_pair.pair, new_token)
            new_token += 1

        print(f"\n{num_merges} merges completed successfully")
        self._is_trained = True

    def _create_pair_heap(self):
        self._token_type_map = {}
        heap = MaxPriorityMap(
            heap_key=lambda x: len(x.positions), map_key=lambda x: x.pair
        )

        pair_heap_items = {}
        for token_list_idx, tokens in enumerate(self._tokens_list):
            for token_idx in range(len(tokens) - 1):
                pair = (tokens[token_idx].value, tokens[token_idx + 1].value)
                # Types: Singular or a Specific Group
                token_type1 = self._token_type_map.setdefault(
                    pair[0], self._get_token_type(self._translate_token(pair[0]))
                )
                token_type2 = self._token_type_map.setdefault(
                    pair[1], self._get_token_type(self._translate_token(pair[1]))
                )

                if not self.is_mergeable_token_types(
                    token_type1, token_type2
                ) or self._is_unmergeable(self._translate_pair(pair)):
                    continue

                if pair not in pair_heap_items:
                    pair_heap_items[pair] = PairHeapItem(pair=pair, positions=set())
                pair_heap_items[pair].positions.add((token_list_idx, token_idx))

        for pair in pair_heap_items:
            heap.push(pair_heap_items[pair])

        print("Heap created successfully\n")

        return heap

    def _merge_pair_and_update_heap(self, merge_pair, new_token):
        for merge_position in list(merge_pair.positions):
            tokens_list_idx, tokens_idx = merge_position
            if merge_position not in merge_pair.positions:
                continue

            # Get the tokens where the merge is happening
            tokens = self._tokens_list[tokens_list_idx]
            self._update_heap_before_merge(
                tokens, merge_pair, merge_position, new_token
            )
            tokens.merge_pair(tokens_idx, new_token)
            merge_pair.positions.remove((tokens_list_idx, tokens_idx))

        print(
            f"({self._vocab.inv.get(merge_pair.pair[0], merge_pair.pair[0])}, "
            f"{self._vocab.inv.get(merge_pair.pair[1], merge_pair.pair[1])})"
            f" -> {new_token} "
        )

    def _update_heap_before_merge(self, tokens, merge_pair, merge_position, new_token):
        # Position of the first token in the merge pair
        tokens_list_idx, tokens_idx = merge_position[0], merge_position[1]
        # Indexes of the tokens right, left and second left of token
        left_idx = tokens.get_previous_idx(tokens_idx)
        right_idx = tokens.get_second_next_idx(tokens_idx)
        second_left_idx = tokens.get_second_previous_idx(tokens_idx)
        # The previous pair of tokens before the merge
        prev_pair_left = prev_pair_right = None
        # Let the new merge token have token type of the second token in the pair
        merge_pair_token_type = self._token_type_map[merge_pair.pair[1]]
        # Get token types of surrounding tokens
        token_type_left = self._get_token_type_from_map(tokens, left_idx)
        token_type_right = self._get_token_type_from_map(tokens, right_idx)
        token_type_second_left = self._get_token_type_from_map(tokens, second_left_idx)
        # Check if the new token can merge with it's surroundings
        left_mergeable, right_mergeable = self._is_new_token_mergeable(
            merge_pair_token_type,
            token_type_left,
            token_type_right,
            token_type_second_left,
        )

        # If a token can never merge left or right it must be a new singular token with
        # no surrounding group
        new_token_type = (
            merge_pair_token_type if left_mergeable or right_mergeable else "Singular"
        )
        # Update token type map
        self._token_type_map[new_token] = new_token_type

        # If the new token type is singular check if it can now merge left or right
        left_mergeable = left_mergeable or (
            new_token_type == token_type_left == "Singular"
        )
        right_mergeable = right_mergeable or (
            new_token_type == token_type_right == "Singular"
        )

        # Update left pair positions on heap
        if left_idx is not None:
            prev_pair_left = (tokens[left_idx].value, merge_pair.pair[0])
            self._remove_position_from_pair(
                merge_pair, prev_pair_left, tokens_list_idx, left_idx
            )
            merged_left = (tokens[left_idx].value, new_token)
            # Merge only similar token types and don't merge unmergeable
            if left_mergeable and not self._is_unmergeable(
                self._translate_pair(merged_left)
            ):
                self._add_position_to_pair(merged_left, tokens_list_idx, left_idx)

        # Update right pair positions on heap
        if right_idx is not None:
            next_idx = tokens.get_next_idx(tokens_idx)
            prev_pair_right = (merge_pair.pair[1], tokens[right_idx].value)
            self._remove_position_from_pair(
                merge_pair, prev_pair_right, tokens_list_idx, next_idx
            )
            merged_right = (new_token, tokens[right_idx].value)
            if right_mergeable and not self._is_unmergeable(
                self._translate_pair(merged_right)
            ):
                # Merge only similar token types and don't merge unmergeable
                self._add_position_to_pair(merged_right, tokens_list_idx, tokens_idx)

    def _is_new_token_mergeable(
        self,
        merged_token_type,
        token_type_left,
        token_type_right,
        token_type_second_left,
    ):
        left_mergeable = self.is_mergeable_token_types(
            token_type_left,
            merged_token_type,
            token_type_previous=token_type_second_left,
            is_merged_token=True,
        )

        right_mergeable = self.is_mergeable_token_types(
            merged_token_type,
            token_type_right,
            token_type_previous=token_type_left,
            is_merged_token=True,
        )

        return left_mergeable, right_mergeable

    def _get_token_type_from_map(self, tokens, idx):
        return self._token_type_map[tokens[idx].value] if idx is not None else None

    def _remove_position_from_pair(self, merge_pair, pair, tokens_list_idx, tokens_idx):
        # Since merge pair hasn't been added to the heap yet check if the pair to remove this the same
        if merge_pair.pair == pair:
            merge_pair.positions.remove((tokens_list_idx, tokens_idx))
            return

        if pair not in self._pair_heap:
            return

        # Remove the pair, remove the position and add it back if needed
        pair_item = self._pair_heap.pop_by_map_key(pair)

        # Sometimes token pairs may be unmergeable at different positions
        # So a unmergeable position will not be in the pair's positions
        if (tokens_list_idx, tokens_idx) in pair_item.positions:
            pair_item.positions.remove((tokens_list_idx, tokens_idx))

        if pair_item.positions:
            self._pair_heap.push(pair_item)

    def _add_position_to_pair(self, pair, tokens_list_idx, tokens_idx):
        pair_item = (
            self._pair_heap.pop_by_map_key(pair)
            if pair in self._pair_heap
            else PairHeapItem(pair=pair, positions=set())
        )
        pair_item.positions.add((tokens_list_idx, tokens_idx))

        self._pair_heap.push(pair_item)

    def _translate_pair(self, pair):
        return tuple(self._vocab.inv.get(token, "") for token in pair)

    def _translate_token(self, token):
        return self._vocab.inv.get(token, "")

    # Default token type getter: every token is Singular
    def _get_token_type(self, token):
        return "Singular"

    def _is_unmergeable(self, pair):
        return False

    def is_mergeable_token_types(self, token_type1, token_type2):
        return token_type1 == token_type2
