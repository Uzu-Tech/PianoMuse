import config
from model.tokenizers.midi_tokenizer import MIDI_Tokenizer, TokenizerTrainer
from model.tokenizers.SONIC import decoder, encoder, token_functions, vocab, score_filterer
from model.utils.midi_preprocessor import MIDIProcessor

if __name__ == "__main__":
    vocab_dict = vocab.build_vocab(config.GENRES)

    processor = MIDIProcessor(config.DATA_DIR, config.ERROR_DIR)

    trainer = TokenizerTrainer(
        vocab_dict,
        config.VOCAB_SIZE,
        token_functions.get_token_type,
        token_functions.is_unmergeable,
        token_functions.is_mergeable_token_types,
    )

    tokenizer = MIDI_Tokenizer(
        processor,
        encoder.Encoder(vocab_dict),
        decoder.Decoder(vocab_dict),
        vocab_dict,
        trainer,
        score_filterer.is_valid_score
    )

    tokenizer.process_and_encode(config.SAVE_DIR, config.MERGES_DIR)

