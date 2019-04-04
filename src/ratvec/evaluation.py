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
from ratvec.eval_utils import plos_cross_val_score
from ratvec.utils import make_balanced, make_ratvec, secho

__all__ = [
    'main',
]


@click.command()
@click.option('-l', '--family-labels', type=click.File('r'), help='Path to family labels', required=True)
@click.option('-d', '--directory', required=True, help='Path to output directory')
@click.option('-c', '--n-components', type=int, default=100)
@click.option('-n', '--max-neighbors', type=int, default=15)
@click.option('--n-iterations', type=int, default=100)
@click.option('--no-save-dataset', is_flag=True)
@click.option('--load-dataset', is_flag=True)
def main(
        family_labels,
        directory,
        n_components: int,
        max_neighbors: int,
        n_iterations: int,
        no_save_dataset: bool,
        load_dataset: bool,
) -> None:
    """Evaluate KPCA embeddings."""
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        if load_dataset:
            click.echo('Loading balanced datasets')
            subdirectory = os.path.dirname(family_labels.name)
            with open(family_labels.name + "_balanced", "rb") as file:
                balanced_datasets, counts = pickle.load(file)
                _sub_run_evaluation(
                    balanced_datasets=balanced_datasets,
                    counts=counts,
                    n_components=n_components,
                    n_iterations=n_iterations,
                    max_neighbors=max_neighbors,
                    pool=pool,
                    subdirectory=subdirectory,
                )
        else:
            secho(f'Loading family labels file: {family_labels}')
            y = np.array([
                l[:-1]
                for l in family_labels
            ])

            optim_dir = os.path.join(directory, 'optim')
            os.makedirs(optim_dir, exist_ok=True)

            secho(f'Dynamically generating balanced datasets from {optim_dir}')
            for subdirectory_name in os.listdir(optim_dir):
                subdirectory = os.path.join(optim_dir, subdirectory_name)
                if not os.path.isdir(subdirectory):
                    continue
                secho(f'Handling {subdirectory}')
                _run_evaluation(
                    y=y,
                    save_dataset=(not no_save_dataset),
                    family_labels=family_labels,
                    n_components=n_components,
                    n_iterations=n_iterations,
                    max_neighbors=max_neighbors,
                    pool=pool,
                    subdirectory=subdirectory,
                )

    secho(f"done. Enjoy your {make_ratvec(3)}")


def _run_evaluation(
        *,
        y,
        save_dataset,
        family_labels,
        n_components,
        n_iterations,
        max_neighbors,
        pool,
        subdirectory,
) -> None:
    kpca = os.path.join(subdirectory, 'kpca.npy')
    secho(f'Loading embeddings file: {kpca}')
    x = np.load(kpca)

    balanced_datasets, counts = make_balanced(x, y)
    if save_dataset:
        family_labels_balanced_path = os.path.join(subdirectory, family_labels.name + "_balanced")
        with open(family_labels_balanced_path, "wb") as file:
            pickle.dump((balanced_datasets, counts), file)

    _sub_run_evaluation(
        balanced_datasets=balanced_datasets,
        counts=counts,
        n_components=n_components,
        n_iterations=n_iterations,
        max_neighbors=max_neighbors,
        pool=pool,
        subdirectory=subdirectory,
    )


def _sub_run_evaluation(
        *,
        balanced_datasets,
        counts,
        n_components,
        n_iterations,
        max_neighbors,
        pool,
        subdirectory,
):
    with open(os.path.join(subdirectory, 'evaluation_params.json'), 'w') as file:
        json.dump(
            dict(
                components=n_components,
                iterations=n_iterations,
                max_neighbors=max_neighbors,
            ),
            file,
            indent=2,
        )

    filt_counts = [
        family_size
        for family_size in counts
        if family_size >= 10
    ]

    secho("Exploring different number of components")
    number_components_grid_search_results = {}
    number_components_low = 1
    number_components_high = int(n_components)
    it = tqdm(
        range(
            number_components_low,
            number_components_high,
            max(1, int(np.floor((number_components_high - number_components_low) / n_iterations))),
        ),
        desc=f'{EMOJI} Optimizing number of components',
    )
    it.write('Number Components\tMean CV Score')
    for reduced_n_components in it:
        n_neighbors = 1
        partial_eval_function = partial(
            plos_cross_val_score,
            reduced_n_components,
            n_neighbors,
        )
        plos_scores = np.array(pool.starmap(partial_eval_function, balanced_datasets))
        weighted_score = np.dot(plos_scores, filt_counts) / np.sum(filt_counts)

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
            plos_cross_val_score,
            best_number_components,
            n_neighbors,
        )
        plos_scores = np.array(pool.starmap(partial_eval_function, balanced_datasets))
        weighted_score = np.dot(plos_scores, filt_counts) / np.sum(filt_counts)

        it.write(f"{n_neighbors}\t{weighted_score:.3f}\b")
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
