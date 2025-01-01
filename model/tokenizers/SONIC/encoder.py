from bisect import bisect_right
from collections import defaultdict
from fractions import Fraction
from functools import partial
from typing import List

from pretty_midi import PrettyMIDI

from model.tokenizers import chords, util

NUM_POSITION_SLOTS = 48


class Encoder:
    def __init__(self, vocab):
        self.all_tokens = []
        self.vocab = vocab
        self.readable = False

    def _reset_states(self):
        self.tokens = []
        self.current_state = defaultdict(lambda: None)
        self.active_notes = {}
        self.sorted_notes = []
        self.chord_notes = []
        self.chord_position = defaultdict(lambda: None)
        self.chord_time = 0
        self.time_to_pos = None
        self.key_signatures = None

    def set_vocab(self, vocab):
        self.vocab = vocab

    def get_number_from_token(self, token):
        if not self.vocab:
            raise ValueError("No vocab found, set vocab first with set_vocab method")

        return self.vocab[token]

    def add_tokens(self, *args: str):
        for token in args:
            if not self.readable:
                self.tokens.append(self.get_number_from_token(token))
            else:
                self.tokens.append(token)

    def change_token(self, idx, new_token):
        if not self.readable:
            self.tokens[idx] = self.get_number_from_token(new_token)
        else:
            self.tokens[idx] = new_token

    def encode(
        self, score: PrettyMIDI, genre: str = "Unknown", readable: bool = False
    ) -> List[str]:
        self.readable = readable
        self._reset_states()
        self._initialize_score_details(score, genre)
        prev_tempo_bpm = prev_ts = prev_key = None
        for note in self.sorted_notes:
            prev_tempo_bpm, prev_ts, prev_key = self.encode_note(
                note, prev_tempo_bpm, prev_ts, prev_key
            )
        self.add_tokens("EOS")
        return self.tokens

    def _initialize_score_details(self, score, genre="Unknown"):
        # Add initial tokens based on key, tempo, and time signature
        self.key_signatures = util.get_key_signatures(score)  # -> List((key, time))
        ts_changes = util.remove_duplicate_time_signatures(score.time_signature_changes)
        tempos_bpm = util.get_score_tempos_bpm(score, ts_changes)

        self.tokens = []
        self.add_tokens("SOS", genre)
        self.sorted_notes = sorted(
            list(util.get_score_notes(score)), key=lambda n: n.start
        )
        self.first_note_time = self.sorted_notes[0].start
        tempo_bar_positions, ts_bar_positions = util.get_time_signature_bar_positions(
            tempos_bpm, ts_changes, self.first_note_time
        )
        self.time_to_pos = partial(
            self._time_to_position,
            tempo_bar_positions=tempo_bar_positions,
            ts_bar_positions=ts_bar_positions,
        )

    # TODO Replace tempo_qpm with bpm
    def _time_to_position(self, time_in_s, tempo_bar_positions, ts_bar_positions):
        ts_times = [ts.time for ts in ts_bar_positions]
        tempo_times = [t.time for t in tempo_bar_positions]
        # Find the current time signature and tempo using binary search
        ts_idx = bisect_right(ts_times, time_in_s) - 1
        time_signature = ts_bar_positions[ts_idx]
        tempo_idx = bisect_right(tempo_times, time_in_s) - 1
        tempo_bpm = tempo_bar_positions[tempo_idx]

        last_bar_change = max(tempo_bpm.bar, time_signature.bar)
        last_time_change = max(tempo_bpm.time, time_signature.time)
        time_between = time_in_s - last_time_change

        num_beats_in_bar = util.get_num_beats_in_bar(time_signature.value)
        bar_position = (
            last_bar_change + (tempo_bpm.value / num_beats_in_bar / 60) * time_between
        )

        current_bar = int(bar_position)
        time_in_beats_after_bar = round(
            (bar_position - current_bar) * num_beats_in_bar, 5
        )  # Rounding here removes floating point errors
        current_beat = int(time_in_beats_after_bar)
        # Number of position slots per beat
        position_in_beat = round(
            (time_in_beats_after_bar - current_beat) * NUM_POSITION_SLOTS
        )
        # Handle beat position overflow
        if position_in_beat >= NUM_POSITION_SLOTS:
            position_in_beat = 0
            current_bar = (
                current_bar + 1
                if current_beat == time_signature.value.numerator
                else current_bar
            )
            current_beat = (current_beat + 1) % time_signature.value.numerator

        # Handle beat overflow
        if Fraction(current_beat / num_beats_in_bar).limit_denominator(12) == 1:
            position_in_beat = 0
            current_beat = 0
            current_bar += 1

        return (
            current_bar,
            current_beat / num_beats_in_bar,
            position_in_beat,
            tempo_bpm.value,
            time_signature.value,
        )

    def encode_note(self, note, prev_tempo_bpm, prev_ts, prev_key):
        # Skip invalid notes
        time_in_s = note.start - self.first_note_time

        note_bar, note_beat, note_pos, active_tempo_bpm, active_ts = self.time_to_pos(
            time_in_s
        )
        active_key = self._get_active_key(note.start)

        if self._is_part_of_chord(note_bar, note_beat, note_pos):
            self.chord_notes.append(note)
            if note != self.sorted_notes[-1]:
                return active_tempo_bpm, active_ts, active_key

        if time_in_s != 0:
            self._add_chord_tokens(
                self.chord_notes,
                self.chord_time,
                prev_tempo_bpm,
                prev_ts,
                prev_key,
            )

        self._update_score_details(active_key, active_tempo_bpm, active_ts)
        self._fix_overlapping_notes(note, time_in_s)
        self._append_position_tokens(note_bar, note_beat, note_pos, active_ts)
        # Add the last note if needed
        if note == self.sorted_notes[-1] and note not in self.chord_notes:
            self.add_tokens("Single")
            # Get the velocity pitch and duration
            self.current_state["velocity"] = self._append_vpod_tokens(
                note,
                self.current_state["velocity"],
                active_tempo_bpm,
                active_ts,
                active_key,
            )

        # Prepare for next note in chord
        (
            self.chord_position["bar"],
            self.chord_position["beat"],
            self.chord_position["pos"],
        ) = (note_bar, note_beat, note_pos)

        self.chord_time = time_in_s
        self.chord_notes = [note]

        return active_tempo_bpm, active_ts, active_key

    def _get_active_key(self, note_start):
        key_idx = bisect_right([k[1] for k in self.key_signatures], note_start) - 1
        return self.key_signatures[key_idx][0]

    def _is_part_of_chord(self, note_bar, note_beat, note_pos):
        return (
            self.chord_position is not None
            and self.chord_position["bar"] == note_bar
            and self.chord_position["beat"] == note_beat
            and self.chord_position["pos"] >= note_pos
            and self.chord_position["pos"] <= note_pos + 3
        )

    def _add_chord_tokens(
        self, chord_notes, chord_time, active_tempo_bpm, active_ts, active_key
    ):
        # Start with the lowest note
        chord_notes = sorted(chord_notes, key=lambda n: n.pitch)
        # Get the chord note pitches
        chord_note_pitches = [chord_note.pitch for chord_note in chord_notes]
        # Recognize the chord from key and pitches
        chord = chords.recognize_chord(self.current_state["key"], chord_note_pitches)
        chord_prefix = "CHORD_" if chord not in ("Single", "Dyad", "Octave") else ""
        self.add_tokens(f"{chord_prefix}{chord}")

        # Then add all the notes in the chord
        for chord_note in chord_notes:
            prev_velocity = self.current_state["velocity"]
            # Get the velocity, pitch, octave and duration
            self.current_state["velocity"] = self._append_vpod_tokens(
                chord_note,
                prev_velocity,
                active_tempo_bpm,
                active_ts,
                active_key,
            )
            # Add the new active notes
            self.active_notes[chord_note.pitch] = {
                "note": chord_note,
                "start": round(chord_time, 6),
                "end": round(chord_time + chord_note.get_duration(), 6),
                "duration_index": len(self.tokens) - 1,
            }

    def _append_vpod_tokens(self, note, prev_velocity, tempo_bpm, time_signature, key):
        v, p, o, d_bars, d_offset = self._get_vpod_tokens(
            note, prev_velocity, tempo_bpm, time_signature, key
        )
        self.add_tokens(
            f"Vel_{v}",
            f"Pitch_{p}",
            f"Octave_{o}",
            f"Duration_{d_bars}brs_{d_offset}off",
        )
        return v if v != "Same" else prev_velocity

    def _get_vpod_tokens(self, note, prev_velocity, tempo_bpm, time_signature, key):
        velocity = note.velocity if note.velocity != prev_velocity else "Same"
        # Normalize pitch based on key
        key = key % 12
        pitch = (note.pitch - key) % 12
        octave = (note.pitch - key - 12) // 12
        octave = util.clamp(octave, 0, 9)

        duration_as_bar_fraction, duration_offset = self._get_duration_tokens(
            note.get_duration(), tempo_bpm, time_signature
        )
        return velocity, pitch, octave, duration_as_bar_fraction, duration_offset

    def _get_duration_tokens(self, duration, tempo_bpm, time_signature):
        num_beats_in_bar = util.get_num_beats_in_bar(time_signature)
        duration_in_beats = round(tempo_bpm / 60 * duration, 6)
        # Limit the duration of a note to 2 bars
        duration_in_beats = min(2 * num_beats_in_bar, duration_in_beats)
        duration_in_beats_int = int(duration_in_beats)
        # Largest denominator is 12
        duration_as_bar_fraction = Fraction(
            duration_in_beats_int / num_beats_in_bar
        ).limit_denominator(12)
        duration_offset = round(
            (duration_in_beats - duration_in_beats_int) * NUM_POSITION_SLOTS
        )
        # Handle beat overflow
        if duration_offset >= NUM_POSITION_SLOTS:
            duration_offset = 0
            duration_in_beats_int = duration_in_beats_int + 1
            duration_as_bar_fraction = Fraction(
                duration_in_beats_int / num_beats_in_bar
            ).limit_denominator(12)

        return duration_as_bar_fraction, duration_offset

    def _update_score_details(self, active_key, active_tempo, active_ts):
        if active_key != self.current_state["key"]:
            self.add_tokens(f"Key_{util.number_to_key(active_key)}")
            self.current_state["key"] = active_key
        if active_tempo != self.current_state["tempo"]:
            self.add_tokens(f"Tempo_{active_tempo}")
            self.current_state["tempo"] = active_tempo
        if active_ts != self.current_state["time signature"]:
            self.add_tokens(
                f"Time Signature_{active_ts.numerator}/{active_ts.denominator}"
            )
            self.current_state["time signature"] = active_ts

    def _fix_overlapping_notes(self, note, time_in_s):
        # Remove notes if they have finished
        active_notes = {
            pitch: values
            for pitch, values in self.active_notes.items()
            if values["end"] > time_in_s
        }
        # Check if the note overlaps with an active note on the same pitch
        if note.pitch in active_notes:
            prev_active_note = active_notes[note.pitch]
            if prev_active_note["end"] > round(time_in_s, 6) and prev_active_note[
                "start"
            ] < round(time_in_s, 6):
                # Adjust the previous note's duration to end before this note starts
                prev_duration_bars, prev_duration_offset = self._get_duration_tokens(
                    time_in_s - prev_active_note["start"],
                    self.current_state["tempo"],
                    self.current_state["time signature"],
                )
                self.change_token(
                    prev_active_note["duration_index"],
                    f"Duration_{prev_duration_bars}brs_{prev_duration_offset}off",
                )

    def _append_position_tokens(self, bar, beat, pos, time_signature):
        num_beats = util.get_num_beats_in_bar(time_signature)
        # Set the bar, beat and position
        if bar != self.current_state["bar"]:
            self.add_tokens("Bar")
            self.current_state["bar"] = bar
        beat_in_bars = Fraction(beat).limit_denominator(12)
        # Add a special token for the last beat
        beat_in_bars = (
            "Last"
            if beat_in_bars.numerator == num_beats - 1
            else beat_in_bars
        )
        self.add_tokens(f"Beat_{beat_in_bars}")
        self.add_tokens(f"Pos_{pos}")
