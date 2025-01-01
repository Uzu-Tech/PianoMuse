import os

import torch

import config
from model.tokenizers.midi_tokenizer import MIDI_Tokenizer
from model.tokenizers.SONIC import decoder, encoder, score_filterer, vocab
from model.transformer.base import RelativeTransformerPredictor
from model.utils.midi_preprocessor import MIDIProcessor
from model.tokenizers.SONIC.vocab import translate_token

# Check if a GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    song_file = os.path.join(config.INSPIRATION_DIR, config.INSPIRATION_SONG_FILENAME)
    inspiration_file = os.path.join(
        config.INSPIRATION_DIR, "inspired " + config.INSPIRATION_SONG_FILENAME
    )
    # Get transformer
    pretrained_filename = os.path.join(
        config.CHECKPOINT_PATH,
        "lightning_logs/version_1/checkpoints",
        config.MODEL_NAME + ".ckpt",
    )
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = RelativeTransformerPredictor.load_from_checkpoint(pretrained_filename)
        model.to(device)

    vocab_dict = vocab.build_vocab(config.GENRES)

    processor = MIDIProcessor(config.DATA_DIR, config.ERROR_DIR)

    tokenizer = MIDI_Tokenizer(
        processor,
        encoder.Encoder(vocab_dict),
        decoder.Decoder(vocab_dict),
        vocab_dict,
        trainer=None,
        is_valid_score=score_filterer.is_valid_score,
    )

    if not os.path.exists(config.MERGES_DIR):
        raise FileNotFoundError(
            "Run midi loader first to train the tokenizer on which tokens"
            "to merge before using model"
        )

    print("Loading Merges...")
    tokenizer.load_merges(config.MERGES_DIR)
    print("Done loading merges\n")

    print("Now encoding file to compose...")
    encoded_tokens = tokenizer.encode_file(
        song_file, config.INSPIRATION_GENRE, readable=False
    )
    print("Done encoding file\n")

    print(
        f"Using first {config.INSPIRATION_STARTING_PERCENTAGE * 100}% of the song as inspiration"
    )
    idx = int(len(encoded_tokens) * config.INSPIRATION_STARTING_PERCENTAGE)
    encoded_tokens = encoded_tokens[:idx]

    print("Creating inspiration...")
    inspired_tokens = get_inspiration(model, encoded_tokens, vocab_dict)

    inspired_score = tokenizer.decode_tokens(inspired_tokens, readable=False)
    inspired_score.save(inspiration_file)


def get_inspiration(model: RelativeTransformerPredictor, encoded_tokens, vocab_dict):
    next_token = None
    inspired_tokens = torch.tensor(encoded_tokens, dtype=torch.long, device=device).unsqueeze(0)
    model.eval()
    print(inspired_tokens)
    print(len(vocab_dict))

    # Iteratively predict next token
    while (
        vocab_dict.inv.get(next_token) != "EOS"
        and inspired_tokens.size(-1) < model.hparams.context_size
    ):
        with torch.no_grad():
            predictions = model.inference(inspired_tokens, temp=1)
            for token in predictions[-20:]:
                print(translate_token(int(token), vocab_dict))
        inspired_tokens = torch.cat((inspired_tokens.squeeze(0), next_token), dim=-1)

    return inspired_tokens

if __name__ == "__main__":
    main()
