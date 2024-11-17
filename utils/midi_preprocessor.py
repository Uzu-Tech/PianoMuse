import os
import pickle
from typing import Dict

from pretty_midi import PrettyMIDI


def preprocess_midi_files(base_directory: str, save_directory: str) -> Dict[str, str]:
    """
    Loads all MIDI files from a specified directory and saves them as PrettyMIDI
    objects dict sorted by genre in a pickle file. Any corrupted files are returned in a
    error dictionary that can be looked at independently.
    """
    midi_files = {}
    error_files = {}
    # Loop through all the genre folders
    genres = os.listdir(base_directory)
    for genre in genres:
        midi_files[genre] = []
        # Skip all non directories
        genre_path = os.path.join(base_directory, genre)
        if not os.path.isdir(genre_path):
            continue
        # Loop through and add all midi files in the directory
        for midi_file in os.listdir(genre_path):
            # Skip all non midi files
            if not midi_file.endswith(".midi") and not midi_file.endswith(".mid"):
                continue
            midi_path = os.path.join(genre_path, midi_file)
            try:
                midi_files[genre].append(PrettyMIDI(midi_path))
            except Exception as e:
                # Errors are outputted to debug corrupted files
                # Count the error files by genre
                error_files[genre] = error_files.get(genre, 0) + 1

    with open(save_directory, "wb") as f:
        pickle.dump(midi_files, f)

    print(f"Score objects saved to {save_directory}")

    return midi_files, error_files


if __name__ == "__main__":
    # Where the midi files are stored
    base_directory = "data"
    # Where you want to store the saved dictionary
    save_directory = "pickle files/pretty midi objects.pkl"
    # The error files counted by genre are printed for debugging
    midi_files, error_files = preprocess_midi_files(base_directory, save_directory)
    print(error_files)
