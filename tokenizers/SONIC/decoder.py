import re
from collections import namedtuple
from dataclasses import dataclass
from fractions import Fraction
from midiutil import MIDIFile
from tokenizers import util
from typing import List

TimeSignature = namedtuple("TimeSignature", ["numerator", "denominator"])

NUM_POSITION_SLOTS = 48


@dataclass
class ScoreState:
    key: int = None
    num_beats_in_bar: int = 0
    time_signature: TimeSignature = TimeSignature(0, 0)
    quarter_notes_since_bar: int = 0  # Placeholder for song hasn't started yet
    beat: int = 0
    pos: int = 0
    velocity: int = 0
    pitch: int = 0
    octave: int = 0
    duration: int = 0


class Decoder:
    def __init__(self) -> None:
        """
        Initializes the Decoder class with a dictionary of token decoders.
        Each token prefix is mapped to the corresponding method for processing.
        """
        self.token_decoders = {
            "Key": self._update_key,
            "Time Signature": self._update_time_signature,
            "Tempo": self._update_tempo,
            "Bar": self._update_bar,
            "Beat": self._update_beat,
            "Pos": self._update_position,
            "Vel_Same": self._skip_velocity,
            "Vel": self._update_velocity_or_pitch_or_octave,
            "Pitch": self._update_velocity_or_pitch_or_octave,
            "Octave": self._update_velocity_or_pitch_or_octave,
            "Duration": self._add_note,
        }
        self._vocab = None
        self.readable = False

    def set_vocab(self, vocab):
        self._vocab = vocab

    def get_token(self, token):
        if self.readable:
            return token
        
        return self._vocab.inverse[token]


    def _reset_states(self) -> None:
        """
        Resets the state of the Decoder and initializes a new MIDIFile.
        """
        self.score = MIDIFile(1, deinterleave=False)
        self.track = 0
        self.channel = 0
        self.score.addTrackName(self.track, time=0, trackName="Piano")
        self.state = ScoreState()
        self.first_bar = self.time_signature_changed = True

    def decode(self, tokens: List[str], readable=False) -> MIDIFile:
        """
        Decodes a list of tokens into a MIDI file.

        Args:
            tokens (List[str]): A list of tokens representing the music score.

        Returns:
            MIDIFile: The generated MIDI file.
        """
        self.readable = readable
        self._reset_states()
        # Find first time signature
        for token in tokens:
            token = self.get_token(token)
            if token.startswith("Time Signature"):
                self._update_time_signature(token)
                break

        for token in tokens:
            token = self.get_token(token)
            # Skip unnecessary tokens
            if self._ignorable_token(token):
                continue

            # Find the appropriate decoder for the token
            for prefix, decode in self.token_decoders.items():
                if token.startswith(prefix):
                    decode(token)  # Call the decode method
                    break

        return self.score

    def save_score(self, score: MIDIFile, filename: str) -> None:
        """
        Saves the generated MIDI file to the specified filename.

        Args:
            score (MIDIFile): The generated MIDI file.
            filename (str): The name of the file to save the MIDI to.
        """
        with open(filename, "wb") as f:
            score.writeFile(f)

    def _ignorable_token(self, token: str) -> bool:
        """
        Determines whether a token represents a non-meaningful MIDI event.

        Args:
            token (str): The token to check.

        Returns:
            bool: True if the token should be ignored, False otherwise.
        """
        return (
            token.startswith("CHORD")
            or token == "Octave"
            or token.startswith("Single")
            or token.startswith("Dyad")
            or token in ("SOS", "EOC")
        )

    # Token decoder methods
    def _add_note(self, token: str) -> None:
        """
        Adds a note to the MIDI file based on the provided token.

        Args:
            token (str): The token representing the note's properties.
        """
        duration = self._get_duration_from_token(token, self.state.num_beats_in_bar)
        # BPM to QPM also converts beats to quarters
        start_time_in_s = self.state.quarter_notes_since_bar + util.bpm_to_qpm(
            (self.state.beat + self.state.pos / NUM_POSITION_SLOTS),
            self.state.time_signature,
            beats_to_quarters=True
        )

        self.score.addNote(
            track=self.track,
            channel=self.channel,
            # Shift by one octave since we shifted down when encoding
            pitch=(self.state.pitch + self.state.key) + ((self.state.octave + 1) * 12),
            # Rounding here fixing floating point errors
            time=round(start_time_in_s, 4),
            duration=round(duration, 4),
            volume=self.state.velocity,
        )

    def _update_key(self, token: str) -> None:
        """
        Updates the key of the score based on the provided token.

        Args:
            token (str): The token representing the key.
        """
        self.state.key = self._get_key_from_token(token)

    def _update_time_signature(self, token: str) -> None:
        """
        Updates the time signature based on the provided token.

        Args:
            token (str): The token representing the time signature.
        """
        time_signature = TimeSignature(*self._get_time_signature_from_token(token))
        if not self.first_bar:
            self._update_bar(None)
        self.state.num_beats_in_bar = util.get_num_beats_in_bar(time_signature)
        self.state.time_signature = time_signature
        self.time_signature_changed = True

    def _update_tempo(self, token: str) -> None:
        """
        Updates the tempo of the score based on the provided token.

        Args:
            token (str): The token representing the tempo.
        """
        tempo_qpm = util.bpm_to_qpm(
            self._get_number_from_token(token), self.state.time_signature
        )
        self.score.addTempo(self.track, self.state.quarter_notes_since_bar, tempo_qpm)

    def _update_bar(self, _: str) -> None:
        """
        Updates the bar count and resets certain time measurements.

        Args:
            _ (str): Placeholder argument, not used in the function.
        """
        if self.first_bar or self.time_signature_changed:  # If it's the start of the song don't add the quarter notes
            self.first_bar = self.time_signature_changed = False
            return
        self.state.quarter_notes_since_bar += (
            self.state.time_signature.numerator / self.state.time_signature.denominator
        ) * 4

    def _update_beat(self, token: str) -> None:
        """
        Updates the beat value based on the provided token.

        Args:
            token (str): The token representing the beat.
        """
        self.state.beat = self._get_beat_from_token(token, self.state.num_beats_in_bar)

    def _update_position(self, token: str) -> None:
        """
        Updates the position of the note based on the provided token.

        Args:
            token (str): The token representing the position.
        """
        self.state.pos = self._get_number_from_token(token)

    def _skip_velocity(self, _: str) -> None:
        """
        Skips processing for "Vel_Same" token.

        Args:
            _ (str): Placeholder argument, not used in the function.
        """
        pass

    def _update_velocity_or_pitch_or_octave(self, token: str) -> None:
        """
        Updates the velocity, pitch, or octave based on the provided token.

        Args:
            token (str): The token representing the velocity, pitch, or octave.
        """
        value = self._get_number_from_token(token)
        if token.startswith("Vel"):
            self.state.velocity = value
        elif token.startswith("Pitch"):
            self.state.pitch = value
        elif token.startswith("Octave"):
            self.state.octave = value

    # Helpers
    def _get_number_from_token(self, token: str) -> int:
        """
        Extracts a number from the token.

        Args:
            token (str): The token containing the number.

        Returns:
            int: The extracted number.
        """
        return int(re.search(r"\d+", token).group())

    def _get_key_from_token(self, key: str) -> int:
        """
        Extracts the key from the token and returns its index in the pitch lookup.

        Args:
            key (str): The token representing the key.

        Returns:
            int: The index of the key in the pitch lookup.
        """
        match = re.search(r"Key_(\w+)", key)
        key = match.group(1)
        return util.pitch_lookup.index(key) % 12

    def _get_time_signature_from_token(self, token: str) -> List[int]:
        """
        Extracts the time signature (numerator and denominator) from the token.

        Args:
            token (str): The token representing the time signature.

        Returns:
            List[int]: A list containing the numerator and denominator of the time signature.
        """
        matches = re.findall(r"\d+", token)
        return [int(match) for match in matches]

    def _get_beat_from_token(self, token: str, num_beats_in_bar: int) -> float:
        """
        Extracts the beat from a token.

        Args:
            token (str): The token representing the beat.
            num_beats_in_bar (int): The number of beats in a bar.

        Returns:
            float: The extracted beat value.
        """
        fraction_pattern = r"(\d+/\d+|\d+|Last)"
        match = re.search(fraction_pattern, token).group()

        if match == "Last":
            return num_beats_in_bar - 1
        return float(num_beats_in_bar * Fraction(match))

    def _get_duration_from_token(self, token: str, num_beats_in_bar: int) -> float:
        """
        Extracts the duration from the token.

        Args:
            token (str): The token representing the duration.
            num_beats_in_bar (int): The number of beats in a bar.

        Returns:
            float: The extracted duration, converted to quarters.
        """
        duration_pattern = r"(\d+/\d+|\d+)"
        duration_in_bars, offset = [
            Fraction(match) for match in re.findall(duration_pattern, token)
        ]
        # Return the duration in quarters
        return float(
            util.bpm_to_qpm(
                duration_in_bars * num_beats_in_bar + offset / NUM_POSITION_SLOTS,
                self.state.time_signature,
                beats_to_quarters=True
            )
        )