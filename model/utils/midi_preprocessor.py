import os
from typing import Dict, List

from pretty_midi import PrettyMIDI

class MIDIProcessor:
    def __init__(self, data_directory: str, error_dir: str = None):
        """
        Initializes the MIDIProcessor class.

        Args:
            data_directory (str): The base directory containing MIDI files or genre folders.
            error_dir (str): Optional path to save a log of corrupted files.
        """
        self.data_directory = data_directory
        self.error_dir = error_dir
        self.midi_files: Dict[str, List[PrettyMIDI]] = {}
        self.error_files: Dict[str, str] = {}

    def _process_genre_folder(self, genre_path: str, genre: str):
        """
        Processes a specific genre folder, loading valid MIDI files.

        Args:
            genre_path (str): Path to the genre folder.
            genre (str): Name of the genre.
        """
        self.midi_files[genre] = []
        for midi_file in os.listdir(genre_path):
            if midi_file.endswith(".mid") or midi_file.endswith(".midi"):
                midi_path = os.path.join(genre_path, midi_file)
                self._load_midi_file(midi_path, genre, midi_file)

    def _load_midi_file(self, midi_path: str, genre: str, midi_file: str):
        """
        Tries to load a MIDI file into PrettyMIDI, handling errors.

        Args:
            midi_path (str): Path to the MIDI file.
            genre (str): Genre of the MIDI file.
            midi_file (str): File name of the MIDI file.
        """
        try:
            if genre not in self.midi_files:
                self.midi_files[genre] = []
            self.midi_files[genre].append(PrettyMIDI(midi_path))
        except Exception as e:
            self.error_files[midi_file] = str(e)

    def process_files(self) -> Dict[str, List[PrettyMIDI]]:
        """
        Processes all MIDI files in the base directory and genre folders.

        Returns:
            Tuple[Dict[str, List[PrettyMIDI]], Dict[str, str]]: A tuple containing
            valid MIDI files and any errors encountered.
        """
        print("Processing midi files...")
        items = os.listdir(self.data_directory)
        for item in items:
            item_path = os.path.join(self.data_directory, item)
            if os.path.isdir(item_path):
                # Process a genre folder
                self._process_genre_folder(item_path, genre=item)
            elif item.endswith(".mid") or item.endswith(".midi"):
                # Process loose MIDI files
                self._load_midi_file(item_path, genre="Unknown", midi_file=item)

        num_scores = sum(len(self.midi_files[genre]) for genre in self.midi_files)
        print(f"{num_scores} Midi files successfully downloaded\n")

        if self.error_dir:
            self._save_error_log()

        return self.midi_files
    
    def _save_error_log(self):
        """
        Saves the corrupted file errors to a specified log file.
        """
        try:
            with open(self.error_dir, "w") as f:
                for key, value in self.error_files.items():
                    f.write(f"{key}: {value}\n")
            print(f"Errors saved to {self.error_dir}")
        except Exception as e:
            print(f"Failed to save error log: {e}")
