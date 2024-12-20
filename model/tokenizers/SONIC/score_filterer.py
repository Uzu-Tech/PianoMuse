from pretty_midi import PrettyMIDI

valid_numerators = [2, 3, 4, 5, 6, 7, 9, 12]
valid_denominators = [2, 4, 8]

def is_valid_score(score: PrettyMIDI):
    if not score.time_signature_changes:
        return False
    for ts in score.time_signature_changes:
        if (
            ts.numerator not in valid_numerators
            or ts.denominator not in valid_denominators
        ):
            return False

    return True
