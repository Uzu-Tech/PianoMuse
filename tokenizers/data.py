import numpy as np

pitch_lookup = [
    "C",
    "C#/Db",
    "D",
    "D#/Eb",
    "E",
    "F",
    "F#/Gb",
    "G",
    "G#/Ab",
    "A",
    "A#/Bb",
    "B",
]

# Define major and minor key templates
major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
major_rolls = np.array([np.roll(major, i) for i in range(12)])
minor_rolls = np.array([np.roll(minor, i) for i in range(12)])
# Stack major and minor rolled distributions
rolled_distributions = np.vstack((major_rolls, minor_rolls))
mean_rolls = rolled_distributions.mean(axis=1)
std_rolls = rolled_distributions.std(axis=1)