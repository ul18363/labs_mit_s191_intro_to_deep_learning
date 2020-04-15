# -*- coding: utf-8 -*-
###############################################################################################
###############################################################################################
###############################################################################################
#%%
# Import Tensorflow 2.0
# %tensorflow_version 2.x
import tensorflow as tf 

# Download and import the MIT 6.S191 package
# !pip install mitdeeplearning
import mitdeeplearning as mdl

# Import all remaining packages
import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm
# !apt-get install abcmidi timidity > /dev/null 2>&1

# Check that we are using a GPU, if not switch runtimes
#   using Runtime > Change Runtime Type > GPU
assert len(tf.config.list_physical_devices('GPU')) > 0
import os
import regex as re
import subprocess
import urllib
import numpy as np
import tensorflow as tf

from IPython.display import Audio
from pathlib import Path
###############################################################################################
###############################################################################################
###############################################################################################
#%%
# cwd = os.path.dirname(__file__)
cwd = os.getcwd()
print(cwd)

def load_training_data():
    cwd = Path(os.getcwd())
    with open(os.path.join(cwd, "data", "irish.abc"), "r") as f:
        text = f.read()
    songs = extract_song_snippet(text)
    return songs


def abc2wav(abc_file):
#     cwd = os.path.dirname(__file__)
    cwd = Path(os.getcwd())
    path_to_tool = os.path.join(cwd, 'bin', 'abc2wav')
    cmd = "{} {}".format(path_to_tool, abc_file)
    return os.system(cmd)

def save_song_to_abc(song, filename="tmp"):
    save_name = "{}.abc".format(filename)
    with open(save_name, "w") as f:
        f.write(song)
    return filename

def extract_song_snippet(text):
    pattern = '\n\n(.*?)\n\n'
    search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
    songs = [song for song in search_results]
    print("Found {} songs in text".format(len(songs)))
    return songs

def play_wav(wav_file):
    return Audio(wav_file)

def play_song(song):
    basename = save_song_to_abc(song)
    ret = abc2wav(basename+'.abc')
    if ret == 0: #did not suceed
        return play_wav(basename+'.wav')
    return None
#%%
songs = load_training_data()

# Print one of the songs to inspect it in greater detail!
example_song = songs[0]
print("\nExample song: ")
print(example_song)
#%%
play_song(example_song)

#%%
# Join our list of song strings into a single string containing all songs
songs_joined = "\n\n".join(songs) 

# Find all unique characters in the joined string
vocab = sorted(set(songs_joined))
print("There are", len(vocab), "unique characters in the dataset")