# -*- coding: utf-8 -*-

"""Evaluation script."""

import json
import pickle
from collections import Counter

import click
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm


@click.command()
@click.option('-l', '--family-labels', type=click.File(), required=True,
              help='Path to family labels')
@click.option('-e', '--kpca', type=click.File('rb'), required=True,
              help='Path to embeddings file')
@click.option('-o', '--output', type=click.File('w'),
              help='Path to output file')
@click.option('-c', '--n-components', default=100, type=int)
@click.option('-n', '--max-neighbors', default=15, type=int)
def main(family_labels, kpca, output, n_components, max_neighbors) -> None:
    """Evaluate and optimize KPCA embeddings."""
    click.echo(f'Loading family labels file: {family_labels}')
    Y = np.array([
        l[:-1]
        for l in family_labels
    ])

    click.echo(f'Loading embeddings file: {kpca}')
    X = pickle.load(kpca)

    clf = KNeighborsClassifier(n_neighbors=1)

    baseline_scores = []
    weights = []
    for family_name in tqdm(set(Y)):
        idx_pos = np.where(Y == family_name)

        family_size = len(idx_pos[0])
        if family_size < 10:
            continue

        X_pos = X[idx_pos]
        idx_neg = np.where(Y != family_name)
        X_neg = X[idx_neg][:family_size]
        X_fam = np.concatenate((X_pos, X_neg), axis=0)
        Y_fam = np.array(family_size * [True] + family_size * [False])

        cv_scores = cross_val_score(clf, X_fam, Y_fam, cv=10, n_jobs=-1)
        mean_cv_score = cv_scores.mean()
        baseline_scores.append(mean_cv_score)
        weights.append(family_size)

    baseline_weighted_score = np.dot(baseline_scores, weights) / np.sum(weights)
    click.echo(f"Baseline setup score: {baseline_weighted_score}")

    label_counter = Counter(Y)
    D = np.array([
        (t, count)
        for t, count in label_counter.items()
        if count >= 10
    ])
    top_labels = [t[0] for t in D]
    idx_top = [l in top_labels for l in Y]

    click.echo("Exploring different number of components")
    number_components_grid_search_results = {}

    start = int(n_components / 3)
    it = tqdm(range(start, n_components))
    it.write('Number Components\tMean CV Score')
    for reduced_n_components in it:
        clf = KNeighborsClassifier(n_neighbors=1)
        cv_scores = cross_val_score(clf, X[idx_top][:, :reduced_n_components], Y[idx_top], cv=10, n_jobs=-1)
        mean_cv_score = cv_scores.mean()
        it.write(f"{reduced_n_components}\t{mean_cv_score}")
        number_components_grid_search_results[reduced_n_components] = mean_cv_score

    best_number_components = max(number_components_grid_search_results, key=number_components_grid_search_results.get)
    best_result1 = number_components_grid_search_results[best_number_components]
    click.echo(f"Best: {best_result1} (N={best_number_components})")

    click.echo("Exploring different number of neighbors")
    number_neighbors_grid_search_results = {}

    it = tqdm(range(1, max_neighbors))
    for n_neighbors in it:
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        cv_scores = cross_val_score(clf, X[idx_top][:, :best_number_components], Y[idx_top], cv=10, n_jobs=-1)
        mean_cv_score = cv_scores.mean()
        it.write("{}\t{}\b".format(n_neighbors, mean_cv_score))
        number_neighbors_grid_search_results[n_neighbors] = mean_cv_score

    best_number_neighbors = max(number_neighbors_grid_search_results, key=number_neighbors_grid_search_results.get)
    best_result2 = number_neighbors_grid_search_results[best_number_neighbors]
    click.echo(f"Best: {best_result2} (N={best_number_neighbors})")

    if output:
        json.dump(
            {
                'baseline': {
                    'average': baseline_weighted_score,
                    'scores': baseline_scores,
                },
                'number_components_grid_search': {
                    'best_number_components': best_number_components,
                    'results': number_components_grid_search_results,
                },
                'number_neighbors_grid_search': {
                    'best_number_neighbors': best_number_neighbors,
                    'results': number_neighbors_grid_search_results,
                },
            },
            output,
            indent=2,
        )


if __name__ == '__main__':
    main()
