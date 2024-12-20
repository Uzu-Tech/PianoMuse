from config import GENRES

def is_unmergeable(tokens):
    return any(
        token.startswith("Time") 
        or token.startswith("Tempo") 
        or token.startswith("Key") 
        or token in ["SOS", "EOC", "EOS"] 
        or token in GENRES
        for token in tokens
    )

def get_token_type(token):
    if token.startswith("Bar") or token.startswith("Beat") or token.startswith("Pos"):
        return "Pos"

    token_types = ["Pitch", "Vel", "Oct", "Dur"]
    for type in token_types:
        if token.startswith(type) and token != "Octave":
            return type

    return "Singular"


def is_mergeable_token_types(
        token_type, token_type_next, token_type_previous=None, is_merged_token=False
    ):
    return (
        (token_type == token_type_next) and (token_type != "Dur" or token_type_previous not in ["Vel", "Pitch", "Oct"])
        or (token_type == "Pitch" and token_type_next == "Oct") 
        or (token_type == "Vel" and token_type_next in ["Oct", "Dur"])
        or (token_type == "Oct" and token_type_next == "Dur" and is_merged_token)
    )