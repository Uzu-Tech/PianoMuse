def recognize_chord(key, note_pitches):
    # Remove duplicates to better recognize chord
    unique_notes = set()
    unique_notes_list = []
    for note in note_pitches:
        if (note % 12) not in unique_notes:
            unique_notes_list.append(note)
            unique_notes.add(note % 12)

    if len(note_pitches) > 1 and len(unique_notes) == 1:
        return "Octave"
    
    note_pitches = unique_notes_list

    if len(note_pitches) < 2:
        return "Single"

    # Put everything in the lowest notes octave
    base_note = min(note_pitches)
    note_pitches = sorted([(pitch - base_note) % 12 for pitch in note_pitches])
    base_note_pitch = base_note % 12
    
    if (chord := get_chord_from_pitches(note_pitches)):
        name, root_note_idx = chord
        root_note = note_pitches[root_note_idx]
        degree = (root_note + base_note_pitch - key) % 12
        return f"{degree}_{name}"
    
    # Any two note interval that isn't a third, fifth or octave is labelled a dyad
    if len(note_pitches) == 2:
        return "Dyad"
    
    # Check for slash chords
    non_base_pitches = note_pitches[1:]
    if len(non_base_pitches) >= 3 and (chord := get_chord_from_pitches(non_base_pitches)):
        name, root_idx = chord
        root_note = non_base_pitches[root_idx]
        degree = (root_note + base_note_pitch - key) % 12
        base_note_degree = (base_note_pitch - key) % 12
        return f"{degree}_{name}/{base_note_degree}"
    
    # Check for extensions be removing all notes except important intervals like thirds
    # and sevenths
    root_note = min(note_pitches)
    degree = (root_note + base_note_pitch - key) % 12
    remaining_note_pitches = [
        note for note in note_pitches
        if note in [0, 3, 4, 7, 10, 11]
    ]

    if (chord := get_chord_from_pitches(remaining_note_pitches)):
        name, root_note_idx = chord
        root_note = note_pitches[root_note_idx]
        degree = (root_note + base_note_pitch - key) % 12
        return f"{degree}_{name} Extension"

    
    return "Tension"

def get_chord_from_pitches(note_pitches):
    intervals = [
        note_pitches[i+1] - note_pitches[i] 
        for i in range(len(note_pitches) - 1)
    ]

    if (chord := chord_map.get(tuple(intervals))):
        return chord
    
    return None

chord_map = {
    # Intervals
    (7, ): ("Pow", 0)   ,          # Perfect fifth (power chord)
    (5, ): ("Pow_Inv", 1),         # Perfect fourth (inverted power chord)

    # Major triads
    (4, ): ("Maj_3rd", 0),        # Major 3rd
    (8, ): ("Maj_3rd_Inv", 1),    # Major 3rd inversion
    (4, 3): ("Maj_R", 0),         # Major root position
    (3, 5): ("Maj_1", 2),         # Major 1st inversion
    (5, 4): ("Maj_2", 1),         # Major 2nd inversion

    # Minor triads
    (3, ): ("Min_3rd", 0),        # Minor 3rd
    (9, ): ("Min_3rd_Inv", 1),    # Minor 3rd inversion
    (3, 4): ("Min_R", 0),         # Minor root position
    (4, 5): ("Min_1", 2),         # Minor 1st inversion
    (5, 3): ("Min_2", 1),         # Minor 2nd inversion

    # Augmented Chords
    (4, 4): ("Aug_R", 0),         # Augmented root position

    # Diminished Chord
    (3, 3): ("Dim_R", 0),         # Diminished root position
    (3, 6): ("Dim_1", 2),         # Diminished 1st inversion
    (6, 3): ("Dim_2", 1),         # Diminished 2nd inversion

    # Major 7th chords
    (4, 3, 4): ("Maj7_R", 0),     # Major 7th root position
    (3, 4, 1): ("Maj7_1", 3),     # Major 7th 1st inversion
    (4, 1, 4): ("Maj7_2", 2),     # Major 7th 2nd inversion
    (1, 4, 3): ("Maj7_3", 1),     # Major 7th 3rd inversion
    (4, 7): ("Maj7_OM5", 0),     # Major 7th Omitted 5
    (7, 1): ("Maj7_OM5_1", 2),     # Major 7th Omitted 5
    (1, 4): ("Maj7_OM5_2", 1),     # Major 7th Omitted 5

    # Minor 7th chords
    (3, 4, 3): ("Min7_R", 0),     # Minor 7th root position
    (4, 3, 2): ("Min7_1", 3),     # Minor 7th 1st inversion
    (3, 2, 3): ("Min7_2", 2),     # Minor 7th 2nd inversion
    (2, 3, 4): ("Min7_3", 1),     # Minor 7th 3rd inversion
    (3, 7): ("Min7_OM5", 0),      # Minor 7th omitted 5th, root position
    (7, 2): ("Min7_OM5_1", 2),    # Minor 7th omitted 5th, 1st inversion
    (2, 3): ("Min7_OM5_2", 1) ,    # Minor 7th omitted 5th, 2nd inversion

    # Dominant 7th chords
    (4, 3, 3): ("Dom7_R", 0),     # Dominant 7th root position
    (3, 3, 2): ("Dom7_1", 3),     # Dominant 7th 1st inversion
    (3, 2, 4): ("Dom7_2", 2),     # Dominant 7th 2nd inversion
    (2, 4, 3): ("Dom7_3", 1),     # Dominant 7th 3rd inversion
    (4, 6): ("Dom7_OM5", 0),      # Dominant 7th omitted 5th, root position
    (6, 2): ("Dom7_OM5_1", 2),    # Dominant 7th omitted 5th, 1st inversion
    (2, 4): ("Dom7_OM5_2", 1),     # Dominant 7th omitted 5th, 2nd inversion

    # Suspended chords
    (2, 5): ("Sus2", 0),          # Suspended 2nd root position
    (5, 2): ("Sus4", 0),          # Suspended 4th root position
    (2, 5, 3): ("Min7_Sus2", 0),
    (5, 2, 3): ("Min7_Sus4", 0),
    (2, 5, 4): ("Maj7_Sus2", 0),
    (5, 2, 4): ("Maj7_Sus4", 0)
}