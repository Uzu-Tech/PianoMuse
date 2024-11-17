from pretty_midi import PrettyMIDI
from typing import List

class Tokenizer():
    def __init__(self):
        self.merges = {} # (int, int) -> int

    @property
    def vocab(self):
        pass

    def train(self, text, vocab_size):
        """
        A method to the train the tokenizer on a dataset of pretty midi files, to learn
        the most common merges it needs to perform.
        """
        raise NotImplementedError

    def encode(self, midi_file: 'PrettyMIDI') -> List[int]:
        """
        A method to encode a pretty midi file into a list of indexed tokens
        """
        raise NotImplementedError

    def decode(self, ids):
        """
        A method to decode a list of indexed tokens into a pretty midi file
        """
        raise NotImplementedError
