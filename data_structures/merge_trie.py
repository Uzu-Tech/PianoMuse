from dataclasses import dataclass, field
from bidict import bidict

@dataclass
class TrieNode:
    value: int = 0
    children: dict = field(default_factory=dict)
    is_end: bool = False


class MergeTrie:
    def __init__(self, vocab: bidict):
        self._root = TrieNode()
        self._merges = {}
        self._vocab = vocab


    def _insert_merge(self, merge_pair, new_token):
        # If a merged token, get its vocab tokens from the merge dictionary
        if merge_pair[0] not in self._vocab.inv:
            tokens1 = self._merges[merge_pair[0]]
        else:
            tokens1 = [merge_pair[0], ]

        if merge_pair[1] not in self._vocab.inv:
            tokens2 = self._merges[merge_pair[1]]
        else:
            tokens2 = [merge_pair[1], ]

        # Combine vocab tokens for new merge
        combined_tokens = tokens1 + tokens2
        self._merges[new_token] = combined_tokens


    def _load_merges(self, merges):
        self._root = TrieNode()
        self._merges = {}

        for token in self._vocab.inv:
            # Initialize trie with token vocabulary
            self._root.children[token] = TrieNode()
            self._root.children[token].value = token

        self._merges = merges
        for merge_token in merges:
            current_node = self._root
            for token in merges[merge_token]:
                if token not in current_node.children:
                    current_node.children[token] = TrieNode()
                current_node = current_node.children[token]
            current_node.value = merge_token
            current_node.is_end = True

    
    def _merge(self, tokens):
        current_node = self._root
        merged_tokens = []
        buffer = []  # Temporary storage for tokens in progress of merging

        for token in tokens:
            if token not in self._vocab.inv:
                raise ValueError("Token is not in vocabulary")

            if token in current_node.children:
                # If the token can continue the merge, move down the trie
                buffer.append(token)
                current_node = current_node.children[token]

                if not current_node.children:  # Leaf node reached
                    merged_tokens.append(current_node.value)
                    buffer = []  # Clear the buffer after merging
                    current_node = self._root
            else:
                # Cannot continue merge; append buffered tokens if any
                if buffer:
                    # If at the end of a merge branch only then can we merge
                    if current_node.is_end:
                        merged_tokens.append(current_node.value)
                    else:
                        merged_tokens.extend(buffer)                        
                    buffer = []
                # Start a new sequence with the current token
                merged_tokens.append(token)
                current_node = self._root

        # Handle any leftover tokens in the buffer
        if buffer:
            merged_tokens.append(current_node.value)

        return merged_tokens




    