# config.py
import os

# Base directories
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)  # Directory where the config file resides
DATA_DIR = "data"

# Name for save folder for processed data
LOADER_DIR = "processed data"
ERROR_DIR = os.path.join(LOADER_DIR, "error.txt")
TOKENS_DIR = os.path.join(LOADER_DIR, "merged_tokens.pkl")
MERGES_DIR = os.path.join(LOADER_DIR, "merges.pkl")

# Inspiration
INSPIRATION_DIR = "inspirations"
INSPIRATION_SONG_FILENAME = "piano_sonata_310_1_(c)oguri.mid"
INSPIRATION_GENRE = "Classical"
INSPIRATION_STARTING_PERCENTAGE = 0.2

# Music genres
GENRES = [
    "Ambient",
    "Blues",
    "Children",
    "Classical",
    "Country",
    "Electronic",
    "Folk",
    "Jazz",
    "Latin",
    "Pop",
    "Rap",
    "Reggae",
    "Religious",
    "Rock",
    "Soul",
    "Soundtracks",
    "Unknown",
    "World",
]

# Tokenizer settings
VOCAB_SIZE = 2**16 - 1  # 65535

# Trainer Settings
CHECKPOINT_PATH = "saved_transformers"
MODEL_NAME = "Pre-trained PianoMuse"

MAX_LENGTH = 3400
USE_ALL_GENRES = True
GENRES_TO_TRAIN = ["Classical",] # Used if use all genres is false

# Trainer Hyperparameters
DATASET_SPLIT_SIZES = (0.8, 0.1, 0.1) # (Train, Validate, Test)
MAX_EPOCHS = 8
BATCH_SIZE = 1
ACCUMULATE_GRADIENT_BATCHES = 64
EMBED_DIM = 256
NUM_ATTENTION_HEADS = 8
NUM_LAYERS = 6
LEARNING_RATE = 0.001
WARMUP = 4000
DROPOUT = 0.1

# Other settings
NUM_DEVICES = 1
GRADIENT_CLIP_VALUE = 5 # Usually between 0.5 and 10