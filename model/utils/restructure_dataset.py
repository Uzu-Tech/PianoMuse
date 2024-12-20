import os
import shutil


def move_midi_files(src_dir: str, dest_dir: str):
    """
    Moves all MIDI files from src_dir to dest_dir, renaming them if necessary
    to avoid overwriting existing files.
    """
    for item in os.listdir(src_dir):
        item_path = os.path.join(src_dir, item)
        # Check if it's a directory
        if os.path.isdir(item_path):
            # Recursively handle subdirectories
            move_midi_files(item_path, dest_dir)
            # After moving files, remove the now-empty directory
            os.rmdir(item_path)  # Remove the directory if it's empty
            
        elif item.endswith(".mid"):
            # Create a new filename if a file with the same name exists
            new_filename = item
            counter = 1
            while os.path.exists(os.path.join(dest_dir, new_filename)):
                # Split the file name and extension
                name, ext = os.path.splitext(item)
                new_filename = f"{name}_{counter}{ext}"
                counter += 1
            shutil.move(item_path, os.path.join(dest_dir, new_filename))


def restructure_dataset(base_dir: str):
    """
    Restructures a dataset by removing all unnecessary sub folders and leaving only the
    genre labelled folders with the files that match them. The data folder must only
    contain midi files and nothing else.
    """
    # Iterate through all genre folders
    for genre in os.listdir(base_dir):
        genre_path = os.path.join(base_dir, genre)
        # Check if it's a directory
        if os.path.isdir(genre_path):
            # Move midi files recursively through the tree
            move_midi_files(genre_path, genre_path)


if __name__ == "__main__":
    base_directory = "data"  # Set your base directory here
    restructure_dataset(base_directory)
