import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pickle

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split

import config
from model.transformer.base import RelativeTransformerPredictor
from model.transformer.dataset import MIDITokensDataset

def main():
    # Load the merged tokens
    if os.path.isfile(config.TOKENS_DIR):
        with open(config.TOKENS_DIR, "rb") as f:
            print("Loading tokens...")
            encoded_tokens = pickle.load(f)
            print("Tokens loaded successfully\n")
    else:
        raise FileNotFoundError(
            "No encoded tokens created, run midi loader script first"
        )

    # Only keep tokens under the max length
    org_len = len(encoded_tokens)
    print(f"Filtering out token sequences over max length of {config.MAX_LENGTH}")
    encoded_tokens = [
        tokens for tokens in encoded_tokens if len(tokens) <= config.MAX_LENGTH
    ]
    print(
        f"{org_len - len(encoded_tokens)} sequences were filtered out, new size is now "
        f"{len(encoded_tokens)}\n"
    )

    # Filter genres if needed
    if not config.USE_ALL_GENRES:
        print("Filtering out token sequences are not in: ", end="")
        print(*config.GENRES_TO_TRAIN)
        encoded_tokens = filter_genres(encoded_tokens, config.GENRES_TO_TRAIN)
        print(f"New size is now {len(encoded_tokens)}")

    model, result, logged_metrics = train_music_transformer(
        *get_data_loaders(encoded_tokens)
    )

    print(result)


def get_data_loaders(encoded_tokens):
    # Create dataset and dataloaders
    dataset = MIDITokensDataset(encoded_tokens, max_length=config.MAX_LENGTH)

    # Define the split sizes
    if sum(config.DATASET_SPLIT_SIZES) != 1:
        raise ValueError("Config split sizes must sum to 1")

    # Set the random seed for reproducibility
    seed = 42  # You can choose any integer
    torch.manual_seed(seed)

    # Split dataset up with no overlap
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, config.DATASET_SPLIT_SIZES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


def train_music_transformer(train_loader, val_loader, test_loader):
    # Create a PyTorch Lightning trainer with generation callback
    root_dir = os.path.join(config.CHECKPOINT_PATH)
    os.makedirs(root_dir, exist_ok=True)

    # Save models based on validation loss
    callbacks = [
        ModelCheckpoint(
            mode="min",
            monitor="val_loss",
            filename=config.MODEL_NAME,
        ),
    ]

    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=callbacks,
        accelerator="auto",
        devices=config.NUM_DEVICES,
        max_epochs=config.MAX_EPOCHS,
        # Clip gradients to prevent exploding gradients
        gradient_clip_val=config.GRADIENT_CLIP_VALUE,
        # Reduces Memory
        precision=16,
        accumulate_grad_batches=config.ACCUMULATE_GRADIENT_BATCHES,
        log_every_n_steps=20
    )
    # Don't need to log hyperparameters right now
    trainer.logger._default_hp_metric = None

    # Check for pretrained model
    pretrained_filename = os.path.join(
        config.CHECKPOINT_PATH,
        #"lightning_logs/version_0/checkpoints",
        config.MODEL_NAME + ".ckpt",
    )
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = RelativeTransformerPredictor.load_from_checkpoint(pretrained_filename)
        trainer.fit(model, train_loader, val_loader)
    else:
        # Create a new model with hyperparameters in config file
        model = RelativeTransformerPredictor(
            config.VOCAB_SIZE,
            config.MAX_LENGTH,
            config.EMBED_DIM,
            config.NUM_ATTENTION_HEADS,
            config.NUM_LAYERS,
            config.LEARNING_RATE,
            config.WARMUP,
            max_iters=trainer.max_epochs * len(train_loader),
            padding_idx=config.VOCAB_SIZE,
            dropout=config.DROPOUT,
        )
        # TODO: Remember to remove ckpt path
        trainer.fit(model, train_loader, val_loader)

    # Test the model on both the validation and testing dataset and note the loss
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {
        "test_loss": test_result[0]["test_loss"],
        "val_loss": val_result[0]["val_loss"],
    }
    return model, result

def collate_fn(batch):
    # Convert to torch tensors
    inputs, labels = zip(*batch)
    inputs = torch.tensor(inputs, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return inputs, labels


def filter_genres(tokens_list, genres):
    genre_tokens = [config.GENRES.index(genre) for genre in genres]
    return [tokens for tokens in tokens_list if tokens[1] in genre_tokens]


if __name__ == "__main__":
    main()
