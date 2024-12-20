from model.tokenizers.chords import chord_map
from model.tokenizers.util import number_to_key
from bidict import bidict
from itertools import product
from fractions import Fraction

def build_vocab(genres):
    vocab = bidict()
    vocab_idx = 0
    # Genre
    for genre in genres:
        vocab[genre] = vocab_idx
        vocab_idx += 1

    # Key
    for i in range(0, 24):
        vocab[f"Key_{number_to_key(i)}"] = vocab_idx
        vocab_idx += 1

    # Tempo
    for i in range(25, 276):
        vocab[f"Tempo_{i}"] = vocab_idx
        vocab_idx += 1
        
    # Time Signature
    numerators = [2, 3, 4, 5, 6, 7, 9, 12]
    denominators = [2, 4, 8]
    for numerator, denominator in product(numerators, denominators):
        vocab[f"Time Signature_{numerator}/{denominator}"] = vocab_idx
        vocab_idx += 1

    # Bar
    vocab["Bar"] = vocab_idx
    vocab_idx += 1
    # Beat
    num_beats = [2, 3, 4, 5, 7]
    for beat_num in num_beats:
        for beat_pos in range(beat_num - 1):
            beat_vocab = f"Beat_{Fraction(beat_pos/beat_num).limit_denominator(12)}"
            if beat_vocab not in vocab:
                vocab[beat_vocab] = vocab_idx
                vocab_idx += 1
    # Beat Last
    vocab["Beat_Last"] = vocab_idx
    vocab_idx += 1

    # Pos
    for i in range(48):
        vocab[f"Pos_{i}"] = vocab_idx
        vocab_idx += 1
    
    # Chords
    for intervals, chord in chord_map.items():
        chord_name = chord[0]
        # Basic chords and Extensions
        for degree in range(12):
            vocab[f"CHORD_{degree}_{chord_name}"] = vocab_idx
            vocab[f"CHORD_{degree}_{chord_name} Extension"] = vocab_idx + 1
            vocab_idx += 2
        # Slash chords
        if len(intervals) >= 2:
            for degree in range(12):
                vocab[f"CHORD_{degree}_{chord_name}_Slash"] = vocab_idx
                vocab_idx += 1

    for chord in ["Single", "Dyad", "Octave", "CHORD_Tension"]:
        vocab[chord] = vocab_idx
        vocab_idx += 1

    # Velocity
    for i in range(128):
        vocab[f"Vel_{i}"] = vocab_idx
        vocab_idx += 1
    
    vocab["Vel_Same"] = vocab_idx
    vocab_idx += 1
            
    # Pitch
    for i in range(12):
        vocab[f"Pitch_{i}"] = vocab_idx
        vocab_idx += 1
    
    # Octave
    for i in range(10):
        vocab[f"Octave_{i}"] = vocab_idx
        vocab_idx += 1

    # Duration
    for beat_num in num_beats:
        for beat_pos in range(beat_num * 2 + 1):
            beat = Fraction(beat_pos/beat_num).limit_denominator(12)
            for pos in range(48):
              duration_token = f"Duration_{beat}brs_{pos}off"
              if duration_token not in vocab:
                  vocab[duration_token] = vocab_idx
                  vocab_idx += 1 

    for token in ("SOS", "EOC", "EOS"):
        vocab[token] = vocab_idx
        vocab_idx += 1

    return vocab


