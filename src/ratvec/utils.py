# -*- coding: utf-8 -*-

import os
from typing import List

import click
import numpy as np
import requests
from tqdm import tqdm

from ratvec.constants import (
    EMOJI,
    METADATA_DOWNLOAD_URL,
    SEQUENCES_DOWNLOAD_URL,
    PROTEIN_FAMILY_SEQUENCES,
    PROTEIN_FAMILY_METADATA
)

__all__ = [
    'ngrams',
    'normalize_word',
    'make_balanced',
    'normalize_kernel_matrix',
    'make_ratvec',
]


def ngrams(s: str, n: int) -> List[str]:
    """Generate n-grams on the given string."""
    string = " " + s + " "
    return [
        string[i:i + n]
        for i in range(len(string) - n + 1)
    ]


def normalize_word(w: str) -> str:
    """Normalize word by uppering cases."""
    return w.upper()


def make_balanced(x, y):
    family_list, inv_ids, counts = np.unique(y, return_inverse=True, return_counts=True)
    balanced_datasets = []
    for family_id in tqdm(range(counts.shape[0]), desc='generating balanced datasets'):
        if counts[family_id] < 10:
            continue
        idx_pos = inv_ids == family_id
        family_size = counts[family_id]
        x_pos = x[idx_pos]
        idx_neg = ~idx_pos
        x_neg = x[idx_neg][:family_size]
        x_fam = np.concatenate((x_pos, x_neg), axis=0)
        y_fam = np.array(family_size * [True] + family_size * [False])
        balanced_datasets.append((x_fam, y_fam))

    return balanced_datasets, counts


def make_ratvec(n: int) -> str:
    """Make a rat vector."""
    if n <= 1:
        return f"<{EMOJI}>"

    return f"<{', '.join(EMOJI * (n - 1))}, ..., {EMOJI}>"


def normalize_kernel_matrix(similarity_matrix):
    """Convert a similarity matrix into a kernel matrix.

    This means that it gets centered and made symmetric.
    """
    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError(f'The given similarity matrix is not square. Actual dimensions: {similarity_matrix.shape}')

    one_n = np.ones_like(similarity_matrix) / similarity_matrix.shape[0]
    return (
            similarity_matrix
            - one_n.dot(similarity_matrix)
            - similarity_matrix.dot(one_n)
            + one_n.dot(similarity_matrix).dot(one_n)
    )


def secho(message, **kwargs) -> None:
    """Wrap :func:`click.secho` with the RatVec emoji."""
    click.secho(message=f"{EMOJI} {message}", **kwargs)


def download_protein_files(directory: str):
    """Download SwissProt protein files."""
    secho(f'Downloading {PROTEIN_FAMILY_SEQUENCES}')
    response = requests.get(SEQUENCES_DOWNLOAD_URL)

    with open(os.path.join(directory, PROTEIN_FAMILY_SEQUENCES), 'wb') as file:
        file.write(response.content)

    secho(f'Downloading {PROTEIN_FAMILY_METADATA}')
    response = requests.get(METADATA_DOWNLOAD_URL)

    with open(os.path.join(directory, PROTEIN_FAMILY_METADATA), 'wb') as file:
        file.write(response.content)
