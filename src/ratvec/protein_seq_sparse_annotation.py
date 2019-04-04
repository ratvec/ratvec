# -*- coding: utf-8 -*-

"""Protein Seq Sparse annotation module."""

import codecs
import os

import numpy as np
from ratvec.constants import DATA_DIRECTORY
from ratvec.utils import ngrams

seq_path = os.path.join(DATA_DIRECTORY, "family_classification_sequences.csv")
metadata_path = os.path.join(DATA_DIRECTORY, "family_classification_metadata.csv")

with open(seq_path) as file:
    # next(file) # throw away the first line rather than using the [1:]
    seqs = [
               set(s[:-1])
               for s in file
           ][1:]

vocab = seqs[0]
for s in seqs:
    vocab |= s

vocab |= {" "}
aminoacid_dict = {}
for a in vocab:
    aminoacid_dict[a] = len(aminoacid_dict.keys())

trigram_dict = {}
for x in vocab:
    for y in vocab:
        for z in vocab:
            trigram_dict[x + y + z] = len(trigram_dict.keys())

with codecs.open(seq_path, "r") as f_in:
    seqs = [s[:-1] for s in f_in][1:]

v_size = len(trigram_dict.keys())

V = np.zeros([len(seqs), v_size], dtype=int)

for i in np.arange(V.shape[0]):
    for t in ngrams(seqs[i], 3):
        V[i, trigram_dict[t]] = 1

np.save("seqs_sparse_trigram_vectors.npy", V)
