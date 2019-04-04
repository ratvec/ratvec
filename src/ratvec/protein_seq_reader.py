# -*- coding: utf-8 -*-

"""Protein Seq Reader module."""

import codecs
import json
import os
import sys
from collections import Counter
from random import shuffle

import click
import numpy as np

from ratvec.constants import (
    DATA_DIRECTORY,
    make_data_directory,
    PROTEIN_FAMILY_METADATA,
    PROTEIN_FAMILY_SEQUENCES
)
from ratvec.utils import download_protein_files, make_ratvec, secho

__all__ = [
    'generate_protein_vocabularies',
    'main',
]


@click.command()
@click.option(
    '-d', '--directory', type=click.Path(dir_okay=True, file_okay=False), required=True,
    help='The output directory for the generated protein vocabularies',
    default=DATA_DIRECTORY
)
@click.option('-f', '--force', is_flag=True)
def main(directory: str, force: bool):
    """Generate the protein vocabularies."""
    # Ensure data directory exists
    make_data_directory()

    sequences_path = os.path.join(directory, PROTEIN_FAMILY_SEQUENCES)
    metadata_path = os.path.join(directory, PROTEIN_FAMILY_METADATA)

    if not force and os.path.isfile(sequences_path) and os.path.isfile(metadata_path):
        secho(f"Files are already existing in {directory}. Use --force to re-compute.")
        sys.exit(0)

    secho(f"Downloading files from the internet. Please be patient.")
    # Download the protein files
    download_protein_files(directory)

    generate_protein_vocabularies(directory, directory)
    secho(f"done. Enjoy your {make_ratvec(3)}")


def generate_protein_vocabularies(source_directory: str, output_directory: str) -> None:
    """Use the data in the source directory to pre-compute files for RatVec."""
    metadata_path = os.path.join(source_directory, PROTEIN_FAMILY_METADATA)
    seq_path = os.path.join(source_directory, PROTEIN_FAMILY_SEQUENCES)
    vocab_file_path = os.path.join(source_directory, "X.txt")
    labels_file_path = os.path.join(source_directory, "Y.txt")

    secho(f'Reading labels from {metadata_path}', fg="cyan")
    with codecs.open(metadata_path) as file:
        _ = next(file)  # skip the header
        # Parse each line to get the protein name
        protein_names = np.array([
            line[:-1].split("\t")[-2]
            for line in file
        ])

    number_of_proteins = len(protein_names)

    # Ensure the number of proteins is 323018
    assert number_of_proteins == 324018, 'Wrong number of protein sequences'

    # Get a list from 0 to number of proteins and shuffle it
    idx = list(range(number_of_proteins))
    shuffle(idx)

    protein_names = protein_names[idx]

    assert len(set(protein_names)) == 7027, 'Wrong number of protein families'

    secho(f'Reading sequences from {seq_path}', fg="cyan")
    with codecs.open(seq_path) as file:
        _ = next(file)  # Skip the header
        seqs = np.array([
            line[:-1]
            for line in file
        ])

    seqs = seqs[idx]

    # Use of a dictionary to remove duplicated sequences
    secho(f'Removing duplicates', fg="cyan")

    secho(f'Number of sequences before removing duplicates {number_of_proteins}', fg="cyan")

    dataset = {
        seqs[i]: protein_names[i]
        for i in idx
    }

    secho(f'Number of sequences after removing duplicates {len(dataset)}', fg="cyan")

    # Free up memory
    del seqs
    del protein_names

    # Get the keys and values of the cleaned up dictionary with no duplicates
    x, y = np.array(list(dataset.keys())), np.array(list(dataset.values()))

    secho(f'Saving vocabulary to {vocab_file_path}', fg="cyan")
    with codecs.open(vocab_file_path, "w") as file:  # Store the sequences into X.txt
        file.write("\n".join(x))

    secho(f'Saving labels to to {labels_file_path}', fg="cyan")
    with codecs.open(labels_file_path, "w") as file:
        file.write("\n".join(y))

    # Make a counter to get the representative vocabularies
    label_counter = Counter(y)
    d = np.array([
        (key, label_counter[key])
        for key in label_counter
    ])
    d_sorted = sorted(d, key=lambda tup: float(tup[1]))

    preset_lengths = 100, 200, 500, 1000, 2000, 3000, 4000
    length_to_subdirectory = {
        length: os.path.join(output_directory, str(length))
        for length in preset_lengths
    }
    length_to_subdirectory[len(set(y))] = os.path.join(output_directory, 'full')

    with open(os.path.join(output_directory, 'manifest.json'), 'w') as file:
        json.dump(
            [
                dict(length=length, subdirectory=subdirectory)
                for length, subdirectory in length_to_subdirectory.items()
            ],
            file,
            indent=2,
        )

    secho(f'Processing for lengths: {", ".join(map(str, sorted(length_to_subdirectory)))}')
    for length, subdirectory in length_to_subdirectory.items():
        os.makedirs(subdirectory, exist_ok=True)
        secho(f'Processing top {length} in {subdirectory}', fg="cyan")

        top_labels = [t[0] for t in d_sorted[-length:]]
        idx_top = [l in top_labels for l in y]

        y_top = y[idx_top]
        x_top = x[idx_top]

        top_n_labels_path = os.path.join(subdirectory, "labels.txt")
        with codecs.open(top_n_labels_path, "w") as file:
            file.write("\n".join(y_top))

        top_n_vocab_path = os.path.join(subdirectory, "full_vocab.txt")
        with codecs.open(top_n_vocab_path, "w") as file:
            file.write("\n".join(x_top))

        single_family_representatives = []
        for family in top_labels:
            family_idx = np.where(y == family)
            # FIXME why does this look for the index, then just get the value? Why not just do
            #  min(x[family_idx], key=len)
            repr_idx = np.argmin([len(s) for s in x[family_idx]])
            single_family_representatives.append(x[family_idx][repr_idx])

        repr_top_n_path = os.path.join(subdirectory, "repr_vocab.txt")
        with codecs.open(repr_top_n_path, "w") as file:
            file.write("\n".join(single_family_representatives))


if __name__ == '__main__':
    main()
