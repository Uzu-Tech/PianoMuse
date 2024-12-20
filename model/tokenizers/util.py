from bisect import bisect_right
from collections import namedtuple
from functools import partial
from typing import Callable, Iterator

import numpy as np
from pretty_midi import Note, PrettyMIDI, TimeSignature

from model.tokenizers.data import (
    mean_rolls,
    pitch_lookup,
    rolled_distributions,
    std_rolls,
)


def get_scores(midi_files: dict[str, list["PrettyMIDI"]]) -> Iterator[object]:
    """
    Generator function that yields each score from a dictionary of MIDI files grouped by genre.

    Args:
        midi_files (dict): A dictionary where keys are genre names and values are lists of MIDI score objects.

    Yields:
        score: A MIDI score object from the input dictionary.
    """
    for genre in midi_files:
        for score in midi_files[genre]:
            yield score


def get_score_notes(score: "PrettyMIDI") -> Iterator["Note"]:
    """
    Generator function that yields each note from all instruments in a given score.

    Args:
        score (object): A MIDI score object containing multiple instruments.

    Yields:
        note: A note object from the instruments in the score.
    """
    for instrument in score.instruments:
        for note in instrument.notes:
            yield note


def number_to_key(num: int) -> str:
    """
    Converts a numerical representation of a key to its string name with tonality.

    Args:
        num (int): The numerical key representation, where the modulo 12 value corresponds to the pitch,
                   and the division by 12 indicates the tonality (0 for Major, 1 for Minor).

    Returns:
        str: The key name, e.g., "C Major" or "A Minor", based on the input number.
    """
    key = pitch_lookup[num % 12]  # Map the pitch number to the corresponding key
    tonality = num // 12  # Determine the tonality (Major or Minor)
    return (
        f"{key} Major" if tonality == 0 else f"{key} Minor"
    )  # Return the key with its tonality


def get_key_signatures(score: "PrettyMIDI") -> list[tuple[int, float]]:
    """
    Extracts key signatures from a score, returning a list of key numbers and their corresponding times.

    Args:
        score (object): A MIDI score object with key signature changes.

    Returns:
        list: A list of tuples, each containing a key number and the time at which the key change occurs.
              If no key signatures are found, one is calculated from the score
    """
    return [
        (key.key_number, key.time)
        for key in remove_duplicate_key_signatures(score.key_signature_changes)
    ] or [
        (get_key(score), 0.0)
    ]  # Calculate key signature if none was found


def get_key(score: "PrettyMIDI") -> str:
    """
    Estimates the key of a score based on its pitch class histogram.

    Args:
        score (object): A MIDI score object with pitch data.

    Returns:
        str: The estimated key of the score, determined from the pitch distribution.
    """
    # Get pitch distribution, factoring in note duration
    pitch_distribution = score.get_pitch_class_histogram(use_duration=True)
    return guess_key_from_distribution(pitch_distribution)


def guess_key_from_distribution(pitch_distribution: np.ndarray) -> int:
    """
    Guesses the key (major or minor) based on the pitch class histogram distribution.

    Args:
        pitch_distribution (numpy.array): A 12-element array representing the pitch class distribution
                                           of a MIDI score, with pitch classes in the range [0, 11].

    Returns:
        int: The index of the guessed key, with values 0-11 for major keys and 12-23 for minor keys.
    """
    # Calculate means and standard deviations for normalization
    mean_dist = pitch_distribution.mean()
    std_dist = pitch_distribution.std()

    # Normalize the distributions and calculate correlation coefficients
    correlations = np.dot(
        rolled_distributions - mean_rolls[:, None], pitch_distribution - mean_dist
    ) / (
        std_rolls * std_dist * len(pitch_distribution)
    )  # Correlation based on normalized pitch distributions

    # Separate major and minor correlations
    major_corrs = correlations[:12]  # Correlations for major keys (0-11)
    minor_corrs = correlations[12:]  # Correlations for minor keys (12-23)

    # Determine the best match for major and minor keys
    max_major_index = np.argmax(
        major_corrs
    )  # Find the major key with the highest correlation
    max_minor_index = np.argmax(
        minor_corrs
    )  # Find the minor key with the highest correlation
    max_major_corr = major_corrs[max_major_index]
    max_minor_corr = minor_corrs[max_minor_index]

    if max_major_corr > max_minor_corr:
        return (
            max_major_index  # Return major key index if the major correlation is higher
        )
    else:
        return max_minor_index + 12  # Return minor key index, offset by 12


def remove_duplicates(
    changes: list[object], is_duplicate: Callable[[object, object], bool]
) -> list[object]:
    """
    Removes consecutive duplicate items from a list based on a custom duplicate check.

    Args:
        changes (list): A list of changes where consecutive duplicates need to be removed.
        is_duplicate (function): A function that takes two items and returns True if they are duplicates.

    Returns:
        list: A new list with consecutive duplicates removed.
    """
    if len(changes) <= 1:
        return changes  # No duplicates if the list has one or fewer items

    while True:
        cleaned_changes = [changes[0]]  # Start with the first element
        last_change = changes[0]
        changed = False  # Flag to track if any duplicates are removed

        for change in changes[1:]:
            if not is_duplicate(change, last_change):
                cleaned_changes.append(change)  # Add the item if it's not a duplicate
            else:
                changed = True  # Mark that a duplicate was found and removed
            last_change = change

        if not changed:  # Exit loop if no duplicates were removed in this iteration
            break

        changes = cleaned_changes  # Update the list for the next iteration

    return cleaned_changes


# Define partial functions for each type of removing duplicates
remove_duplicate_key_signatures = partial(
    remove_duplicates,
    is_duplicate=lambda key, last_key: key.key_number == last_key.key_number,
)
remove_duplicate_time_signatures = partial(
    remove_duplicates,
    is_duplicate=lambda ts, last_ts: ts.numerator == last_ts.numerator
    and ts.denominator == last_ts.denominator,
)
remove_duplicate_tempos = partial(
    remove_duplicates,
    # Registering a Tempo change only if it's 5bpm greater or less than the previous
    is_duplicate=lambda tempo, last_tempo: tempo[0] <= last_tempo[0] + 5
    and tempo[0] >= last_tempo[0] - 5,
)


def get_score_tempos_bpm(
    score: "PrettyMIDI", time_signature_changes: list[object]
) -> list[list[int, float]]:
    """
    Retrieves the tempos in beats per minute (BPM) for a given score, adjusting for time signature changes.

    Args:
        score (object): A MIDI score object with tempo changes.
        time_signature_changes (list): A list of time signature changes, each containing a time and time signature.

    Returns:
        list: A list of tuples, each containing a BPM value and the corresponding time for the tempo change.
    """
    ts_times = [ts.time for ts in time_signature_changes]
    times, tempo_values = score.get_tempo_changes()  # (qpm)
    tempos = remove_duplicate_tempos(list(zip(times, tempo_values)))  # -> (time, qpm)

    tempos_bpm = []
    for start, qpm in tempos:
        # Find the appropriate time signature for the given tempo start time using binary search
        ts_idx = bisect_right(ts_times, start) - 1
        time_signature = time_signature_changes[ts_idx]

        # Convert the tempo (qpm) to BPM, adjusted by the time signature
        bpm = round(
            clamp(bpm_to_qpm(qpm, time_signature), 25, 275)
        )  # Ensure BPM is within a valid range
        tempos_bpm.append([bpm, start])

    return tempos_bpm


# Beats Per Minute and Quarters Per Minute
# Compound time registers every third beat as a downbeat
def qpm_to_bpm(qpm: float, time_signature: object) -> int:
    """
    Converts a quarter notes per minute (QPM) value to beats per minute (BPM), adjusting for time signature.

    Args:
        qpm (float): The tempo in quarter notes per minute.
        time_signature (object): A time signature object with numerator and denominator values.

    Returns:
        int: The corresponding BPM value, rounded to the nearest integer, adjusted for time signature.
    """
    bpm = qpm * (
        time_signature.denominator / 4
    )  # Adjust QPM by time signature's denominator
    return (
        round(bpm / 3) if is_compound_time(time_signature) else round(bpm)
    )  # Adjust for compound time if needed


def bpm_to_qpm(
    bpm: float, time_signature: object, beats_to_quarters: bool = False
) -> float:
    """
    Converts a beats per minute (BPM) value to quarter notes per minute (QPM), adjusting for time signature.

    Args:
        bpm (float): The tempo in beats per minute.
        time_signature (object): A time signature object with numerator and denominator values.
        beats_to_quarters (bool): Whether to convert BPM to QPM or beats to quarters.

    Returns:
        float: The corresponding QPM value, adjusted for time signature, or the number of quarters.
    """
    qpm = bpm * (
        4 / time_signature.denominator
    )  # Adjust BPM by time signature's denominator
    if not beats_to_quarters:
        return (
            round(qpm * 3) if is_compound_time(time_signature) else round(qpm)
        )  # Round the value if it's QPM
    else:
        return (
            qpm * 3 if is_compound_time(time_signature) else qpm
        )  # This returns the quarters if using beats to quarters


# Define the named tuples for Tempo_Bars and Time_Signature_Bars
Tempo_Bars = namedtuple("Tempo_Bars", ["value", "bar", "time"])
Time_Signature_Bars = namedtuple("Time_Signature_Bars", ["value", "bar", "time"])


def get_bar_position(
    tempos_bpm_by_bar: list[Tempo_Bars],
    ts_changes_by_bar: list[Time_Signature_Bars],
    time_in_s: float,
) -> float:
    """
    Calculates the position within a bar at a given time based on tempo and time signature changes.

    Args:
        tempos_bpm_by_bar (list of Tempo_Bars): A list of tempo changes, each containing a tempo value (BPM), bar index, and time in seconds.
        ts_changes_by_bar (list of Time_Signature_Bars): A list of time signature changes, each containing the time signature value, bar index, and time in seconds.
        time_in_s (float): The time in seconds at which to calculate the bar position.

    Returns:
        float: The position within the current bar at the given time.
    """
    ts_times = [
        ts.time for ts in ts_changes_by_bar
    ]  # List of time signature change times
    tempo_times = [
        t.bar for t in tempos_bpm_by_bar
    ]  # List of bar indices for tempo changes

    # Find the current time signature and tempo using binary search
    ts_idx = bisect_right(ts_times, time_in_s) - 1
    time_signature = ts_changes_by_bar[ts_idx]
    tempo_idx = bisect_right(tempo_times, time_in_s) - 1
    tempo_bpm = tempos_bpm_by_bar[tempo_idx]

    # Determine the last time and bar change
    last_bar_change = max(tempo_bpm.bar, time_signature.bar)
    last_time_change = max(tempo_bpm.time, time_signature.time)
    time_between = time_in_s - last_time_change  # Time elapsed since last change
    num_beats_in_bar = get_num_beats_in_bar(time_signature.value)

    # Calculate the bar position based on tempo and time signature
    bar_position = (
        last_bar_change + (tempo_bpm.value / num_beats_in_bar / 60) * time_between
    )

    # The time in bars
    return bar_position


def get_time_signature_bar_positions(
    tempos_bpm: list[tuple[float, float]],
    time_signatures: list["TimeSignature"],
    first_note_time: float,
) -> tuple[list[Tempo_Bars], list[Time_Signature_Bars]]:
    """
    Calculates the bar positions based on tempo changes and time signature changes.

    Args:
        tempos_bpm (list of tuples): A list of tuples containing BPM and the corresponding time in seconds.
        time_signatures (list of Time_Signature_Bars): A list of time signature changes, each containing the time signature value, bar index, and time in seconds.
        first_note_time (float): The time in seconds when the first note occurs.

    Returns:
        tuple:
            - tempos_bpm_bar_positions (list of Tempo_Bars): A list of Tempo_Bars objects representing the bar positions for each tempo change.
            - ts_bar_positions (list of Time_Signature_Bars): A list of Time_Signature_Bars objects representing the bar positions for each time signature change.
    """
    # Check if there is only one time signature and tempo
    if len(time_signatures) == 1 and len(tempos_bpm) == 1:
        # If there is just one tempo and time signature, return them with zero bars and time
        time_signature = time_signatures[0]
        bpm, _ = tempos_bpm[0]
        return (Tempo_Bars(bpm, 0, 0),), (Time_Signature_Bars(time_signature, 0, 0),)

    # Initialize counters for bars and times
    total_bars = total_bars_time = 0
    tempo_idx = ts_idx = 1  # Starting indexes for tempos and time signatures

    # Pre-compute the times at which the time signatures and tempos change
    ts_times = [ts.time for ts in time_signatures]
    tempo_times = [tempo[1] for tempo in tempos_bpm]

    last_change_time = (
        0  # Keep track of the last time when either tempo or time signature changed
    )

    prev_tempo_bpm = prev_ts = (
        None  # To store previous tempo and time signature information
    )

    # Initialize lists to hold the bar positions for tempos and time signatures
    tempos_bpm_bar_positions = [
        Tempo_Bars(
            tempos_bpm[0][0], 0, 0.0
        ),  # Start with the first tempo at bar 0, time 0
    ]
    ts_bar_positions = [
        Time_Signature_Bars(
            time_signatures[0], 0, 0.0
        ),  # Start with the first time signature at bar 0, time 0
    ]

    # Iterate through tempos and time signatures to calculate the bar positions
    while tempo_idx < len(tempos_bpm) or ts_idx < len(time_signatures):
        # Get the current time signature and tempo for the respective indices
        ts = time_signatures[
            min(ts_idx, len(time_signatures) - 1)
        ]  # Ensure we don't go out of bounds
        tempo_bpm, tempo_bpm_time = tempos_bpm[
            min(tempo_idx, len(tempos_bpm) - 1)
        ]  # Same for tempo

        # Look for a tempo change
        # or if we have processed all time signatures
        if (ts.time >= tempo_bpm_time and tempo_idx < len(tempos_bpm)) or ts_idx >= len(
            time_signatures
        ):
            # Find the closest time signature to the current tempo change time
            ts_at_tempo_idx = bisect_right(ts_times, tempo_bpm_time) - 1
            ts_at_tempo = time_signatures[ts_at_tempo_idx]
            num_beats_in_bar = get_num_beats_in_bar(
                ts_at_tempo
            )  # Get beats in the bar for this time signature

            # Set previous tempo for later bar calculation
            prev_tempo_bpm, _ = tempos_bpm[min(tempo_idx, len(tempos_bpm) - 1) - 1]

            # Calculate how many new bars have occurred since the last change
            time_in_between = tempo_bpm_time - last_change_time
            new_bars = round(
                (prev_tempo_bpm / (num_beats_in_bar * 60)) * time_in_between
            )

            # Update the total bars and total time
            total_bars += new_bars
            total_bars_time += new_bars * ((num_beats_in_bar * 60) / prev_tempo_bpm)

            # Add a new Tempo_Bars entry to keep track of tempo changes in relation to bars
            tempos_bpm_bar_positions.append(
                Tempo_Bars(tempo_bpm, total_bars, total_bars_time - first_note_time)
            )

            # Move to the next tempo
            tempo_idx += 1
            last_change_time = (
                tempo_bpm_time  # Update the last change time to the current tempo time
            )

        else:
            # If the time signature change happens first, calculate bar positions based on that
            tempo_at_ts_idx = bisect_right(tempo_times, tempo_bpm_time) - 1
            tempo_at_ts = tempos_bpm[tempo_at_ts_idx][
                0
            ]  # Get the tempo at the closest time signature change
            prev_ts = time_signatures[min(ts_idx, len(time_signatures) - 1) - 1]
            num_beats_in_bar = get_num_beats_in_bar(
                prev_ts
            )  # Get beats in the bar for this time signature

            # Calculate the time elapsed since the last change
            time_in_between = ts.time - last_change_time

            # Calculate the new bars based on the elapsed time and the current tempo
            new_bars = round((tempo_at_ts / num_beats_in_bar / 60) * time_in_between)

            # Update the total bars and total time
            total_bars += new_bars
            total_bars_time += new_bars * ((num_beats_in_bar * 60) / tempo_at_ts)

            # Add a new Time_Signature_Bars entry to track the bar positions for time signature changes
            ts_bar_positions.append(
                Time_Signature_Bars(ts, total_bars, total_bars_time - first_note_time)
            )

            # Move to the next time signature
            ts_idx += 1
            last_change_time = (
                ts.time
            )  # Update the last change time to the current time signature time

        # Continue the loop until both tempos and time signatures have been processed

    # Return the final bar positions for both tempos and time signatures
    return tempos_bpm_bar_positions, ts_bar_positions


def remove_duplicates_in_front(entries: list[object]) -> list[object]:
    """
    Removes duplicate entries based on the bar number, keeping only the last entry for each bar.

    Args:
        entries (list of objects): A list of entries, where each entry is expected to have a `bar` attribute.

    Returns:
        list: A list of entries with duplicates removed, sorted by the bar number.
    """
    bar_map = {}
    # Iterate through the entries and update the dictionary with the latest value for each bar number
    for item in entries:
        bar_map[item.bar] = item
    # Reconstruct the list, sorted by the original order of bar numbers in `entries`
    result = [bar_map[bar] for bar in sorted(bar_map.keys())]
    return result


def get_num_beats_in_bar(time_signature: "TimeSignature") -> int:
    """
    Calculates the number of beats in a bar based on the time signature.

    Args:
        time_signature (TimeSignature): A time signature object containing the numerator and denominator.

    Returns:
        int: The number of beats in a bar.
    """
    return (
        time_signature.numerator / 3
        if is_compound_time(time_signature)
        else time_signature.numerator
    )


def is_compound_time(time_signature: "TimeSignature") -> bool:
    """
    Determines if a time signature is compound (e.g., 6/8, 9/8, 12/8).

    Args:
        time_signature (TimeSignature): A time signature object containing the numerator.

    Returns:
        bool: True if the time signature is compound, False otherwise.
    """
    return time_signature.numerator in (6, 9, 12)


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamps a value within a specified range.

    Args:
        value (float): The value to be clamped.
        min_value (float): The minimum allowable value.
        max_value (float): The maximum allowable value.

    Returns:
        float: The clamped value, which is within the specified range.
    """
    return max(min(value, max_value), min_value)
