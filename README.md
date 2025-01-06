# PianoMuse

PianoMuse is an AI transformer model designed to generate piano melodies for practice in sight-reading and ear training. This project is still in progress, and the AI model is currently being trained.

## Features

- **SONIC Tokenizer**: The tokenizer breaks down MIDI files into understandable chunks, including chords, pitches, bars, and beats. It applies the Systemic Ordering of Notes, Intervals, and Chords (SONIC) method to process musical data effectively.
- **Model Folder**: Contains the core scripts for the tokenizer, transformer model, and utility functions to load and preprocess data.
- **Scripts**:
  - **midi_loader.py**: Loads MIDI data for training and testing.
  - **trainer.py**: Handles model training.
  - **compose.py**: Tests the trained model by generating melodies.
  - **config.py**: For user to change training parameters and song to compose over.

## Current Status

**DO NOT USE YET**. The AI model is actively being trained. 
