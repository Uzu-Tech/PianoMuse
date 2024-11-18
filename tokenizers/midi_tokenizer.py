from pretty_midi import PrettyMIDI
from typing import List

class MIDI_Tokenizer():
    def __init__(self, encoder, decoder):
        self.merges = {} # (int, int) -> int
        self.encoder = encoder
        self.decoder = decoder

    def build_vocab(self):
        raise NotImplementedError
    
    def encode(self, score: 'PrettyMIDI', readable=False):
        return self.encoder.encode(score, readable)

    def decode(self, score: 'PrettyMIDI', readable=False):
        return self.decoder.decode(score, readable)

    def train(self, text, vocab_size):
        """
        A method to the train the tokenizer on a dataset of pretty midi files, to learn
        the most common merges it needs to perform.
        """
        raise NotImplementedError

