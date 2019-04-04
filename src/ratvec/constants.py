# -*- coding: utf-8 -*-

"""Constants module."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

DATA_DIRECTORY = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, 'data'))
RESULTS_DIRECTORY = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, 'results'))

"""Protein Files"""

PROTEIN_FAMILY_METADATA = 'family_classification_metadata.csv'
DEFAULT_PROTEIN_FAMILY_METADATA = os.path.join(DATA_DIRECTORY, PROTEIN_FAMILY_METADATA)

METADATA_DOWNLOAD_URL = 'https://dataverse.harvard.edu/api/access/datafile/2712444?format=tab&gbrecs=true'

PROTEIN_FAMILY_SEQUENCES = 'family_classification_sequences.csv'
DEFAULT_PROTEIN_FAMILY_SEQUENCES = os.path.join(DATA_DIRECTORY, PROTEIN_FAMILY_SEQUENCES)

SEQUENCES_DOWNLOAD_URL = 'https://dataverse.harvard.edu/api/access/datafile/2712443?format=original&gbrecs=true'

EMOJI = '🐀'


def make_data_directory():
    """Ensure that the data directory exists."""
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
