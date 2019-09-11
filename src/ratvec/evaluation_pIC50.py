# -*- coding: utf-8 -*-

"""Evaluation script."""

import json
import multiprocessing
import os
import pickle
from functools import partial

import click
import numpy as np
from tqdm import tqdm

from ratvec.constants import EMOJI
from ratvec.eval_utils import knn_cross_val_score
from ratvec.utils import make_balanced, make_ratvec, secho

import pandas as pd

ACTIVITY_THRESHOLD = 7.5


__all__ = [
    'main',
]


@click.command()
@click.option('-r', '--ratvec_directory', required=True, help='Path to output directory (RatVec embeddings)')
@click.option('-a', '--activities_directory', required=True, help='Path to training data (processed.tsv and labels.txt)')
def main(
        ratvec_directory,
        activities_directory,

) -> None:
    """Evaluate KPCA embeddings."""
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        optim_dir = os.path.join(ratvec_directory, 'optim')
        os.makedirs(optim_dir, exist_ok=True)

        smiles_dir = os.path.join(activities_directory, 'labels.txt')
        smiles_list = [l[:-1] for l in open(smiles_dir, "r")]

        activities_dir = os.path.join(activities_directory, 'processed.tsv')
        activities_df = pd.read_csv(activities_dir, sep='\t')




        secho(f'Dynamically generating balanced datasets from {optim_dir}')
        for subdirectory_name in os.listdir(optim_dir):
            subdirectory = os.path.join(optim_dir, subdirectory_name)
            if not os.path.isdir(subdirectory):
                continue
            secho(f'Handling {subdirectory}')
            _run_evaluation(
                smiles_list = smiles_list,
                activities_df = activities_df,
                subdirectory=subdirectory,
                pool=pool
            )

        secho(f"done. Enjoy your {make_ratvec(3)}")


def make_datasets(activities_df, smiles2vec):
    datasets = []
    counts = []
    proteins = []
    for protein in activities_df.NAME.unique():
        protein_df = activities_df[ activities_df["NAME"] == protein]
        y = (protein_df["STANDARD_VALUE"] >= ACTIVITY_THRESHOLD).values
        total_count = len(y)
        pos_count = np.sum(y)
        neg_count = total_count - pos_count
        if pos_count < 10 or neg_count < 10: #Each class at least 10 elements
            continue
        #x = protein_df["SMILES"].replace(smiles2vec).values
        x = np.array([smiles2vec[s] for s in protein_df["SMILES"]])
        proteins.append(protein)
        counts.append(total_count)
        datasets.append((x,y))
        print(f"{pos_count}+\t{neg_count}-\t{pos_count/total_count}")
    return datasets, counts, proteins


def _run_evaluation(
        *,
        smiles_list,
        activities_df,
        subdirectory,
        n_components = 20,
        max_neighbors=11,
        pool,
) -> None:
    kpca = os.path.join(subdirectory, 'kpca.npy')
    secho(f'Loading embeddings file: {kpca}')
    x = np.load(kpca)
    smiles2vec = dict(zip(smiles_list,x))
    del x
    protein_datasets, counts, protein_ids = make_datasets(activities_df, smiles2vec)


    secho("Exploring different number of components")
    number_components_grid_search_results = {}
    number_components_low = 1
    number_components_high = int(n_components)
    it = tqdm(
        range(
            number_components_low,
            number_components_high,
            max(1, int(np.floor((number_components_high - number_components_low) ))),
        ),
        desc=f'{EMOJI} Optimizing number of components',
    )
    it.write('Number Components\tMean CV Score')
    for reduced_n_components in it:
        n_neighbors = 1
        partial_eval_function = partial(
            knn_cross_val_score,
            reduced_n_components,
            n_neighbors,
        )
        knn_scores = np.array(pool.starmap(partial_eval_function, protein_datasets))
        weighted_score = np.dot(knn_scores, counts) / np.sum(counts)

        it.write(f"{reduced_n_components}\t{weighted_score:.3f}")
        number_components_grid_search_results[reduced_n_components] = weighted_score

    best_number_components = max(
        number_components_grid_search_results,
        key=number_components_grid_search_results.get,
    )
    best_result1 = number_components_grid_search_results[best_number_components]
    secho(f"Best at components={best_number_components}, score={best_result1:.3f}")

    secho("Exploring different number of neighbors")
    number_neighbors_grid_search_results = {}

    it = tqdm(range(1, max_neighbors), desc=f'{EMOJI} Optimizing number of neighbors')
    for n_neighbors in it:
        partial_eval_function = partial(
            knn_cross_val_score,
            best_number_components,
            n_neighbors,
        )
        knn_scores = np.array(pool.starmap(partial_eval_function, protein_datasets))
        weighted_score = np.dot(knn_scores, counts) / np.sum(counts)
        it.set_postfix({"weighted score": weighted_score, "n neighbor": n_neighbors})
        number_neighbors_grid_search_results[n_neighbors] = weighted_score

    best_number_neighbors = max(number_neighbors_grid_search_results, key=number_neighbors_grid_search_results.get)
    best_result2 = number_neighbors_grid_search_results[best_number_neighbors]
    secho(f"Best at neighbors={best_number_neighbors}, score={best_result2:.3f}")

    with open(os.path.join(subdirectory, 'evaluation_results.json'), 'w') as file:
        json.dump(
            {
                'number_components_grid_search': {
                    'best_number_components': best_number_components,
                    'results': number_components_grid_search_results,
                },
                'number_neighbors_grid_search': {
                    'best_number_neighbors': best_number_neighbors,
                    'results': number_neighbors_grid_search_results,
                },
            },
            file,
            indent=2,
        )


if __name__ == '__main__':
    main()
