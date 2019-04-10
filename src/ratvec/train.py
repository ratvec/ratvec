# -*- coding: utf-8 -*-

"""PLOS Evaluation module."""

import itertools as itt
import json
import logging
import multiprocessing
import os
import pickle
from functools import partial
from math import ceil
from typing import Any, Callable, List

import Bio.SubsMat.MatrixInfo
import click
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import dok_matrix
from tqdm import tqdm

from ratvec.constants import EMOJI
from ratvec.kernel_utils import KERNEL_TO_PROJECTION
from ratvec.utils import make_ratvec, ngrams, normalize_kernel_matrix, normalize_word, secho

logger = logging.getLogger(__name__)

# TODO expand to multiple by having a nested dictionary from name of parameter to list of possibilities for grid search
#: A dictionary from kernel functions to their *single* hyper-parameter over which to run a grid search
kernels = {
    "linear": [
        0,
    ],
    "rbf": [
        # 0.01,
        # 0.1,
        1.0,
        # 10,
    ],
    "poly": [
        2,
        # 3,
    ],
}

ALLOWED_SIMILARITIES = [
    "ngram_sim",
    # "sorensen_plus",  # TODO where did the implementation for this go
    "ngram_intersec",
    'global-alignment',
]


def _preprocess_vocab_file(f):
    return [
        normalize_word(w[:-1])
        for w in f
        if 1 == len(w[:-1].split())
    ]


@click.command()
@click.option('-f', '--full-vocab-file', type=click.File(), required=True,
              help="A path to a file containing the full vocabulary")
@click.option('-r', '--repr-vocab-file', type=click.File(),
              help="A path to a file containing the representative words - ua subset of your vocabulary if it is too "
                   "large for your memory restrictions")
@click.option('-d', '--output', type=click.Path(file_okay=False, dir_okay=True), required=True,
              help="Output folder for the KPCA embeddings")
@click.option('--n-components', type=int, show_default=True,
              help="Number of principal components of the embeddings. "
                   "If not specified, defaults to 2/3 the size of the representative vocabulary")
@click.option('--sim', type=click.Choice(ALLOWED_SIMILARITIES),
              default="ngram_intersec", show_default=True,
              help="Similarity function: 'ngram_sim' (n-gram similarity),'sorensen_plus' "
                   "(Sørensen–Dice index for n-grams)")
@click.option('--sim-alignment-matrix', type=click.Choice(Bio.SubsMat.MatrixInfo.available_matrices),
              default='blosum62', show_default=True,
              help="Similarity matrix to use with --sim==global-alignment")
@click.option('-n', '--n-ngram', type=int, default=2, show_default=True,
              help="Size of the n-grams when n-gram similarity function is used (default: 2)")
@click.option('--use-gpu', is_flag=True)
@click.option('--processes', type=int, default=multiprocessing.cpu_count(), show_default=True,
              help="Number of processes to be started for computation")
def main(
        full_vocab_file: str,
        repr_vocab_file: str,
        output: str,
        n_components: int,
        sim: str,
        sim_alignment_matrix: str,
        n_ngram: int,
        use_gpu: bool,
        processes: int,
) -> None:
    """Compute KPCA embeddings on a given data set."""
    n = n_ngram  # meh
    output = os.path.abspath(output)
    os.makedirs(output, exist_ok=True)

    full_vocab = _preprocess_vocab_file(full_vocab_file)

    if repr_vocab_file is None:
        repr_vocab = full_vocab
    else:
        repr_vocab = _preprocess_vocab_file(repr_vocab_file)

    params_path = os.path.join(output, 'training_manifest.json')
    secho(f'Outputting training information to {params_path}')
    manifest = dict(
        sim=sim,
        n=n,
        len_full_vocab=len(full_vocab),
        len_repr_vocab=len(repr_vocab),
        kernels=kernels,
    )
    with open(params_path, 'w') as file:
        json.dump(manifest, file, sort_keys=True, indent=2)

    if use_gpu:
        import cudamat as cm
        cm.cublas_init()

    if sim == 'global-alignment':
        secho(f'Computing global alignment similarities with {sim_alignment_matrix}')
        repr_similarity_matrix = calculate_global_alignment_similarity_matrix(
            full_vocab=repr_vocab,
            repr_vocab=repr_vocab,
            processes=processes,
            matrix=sim_alignment_matrix,
            tqdm_desc=f'{EMOJI} Computing self-similarity matrix for '
                      f'repr vocab with global alignment ({sim_alignment_matrix})'
        )
        full_similarity_matrix = calculate_global_alignment_similarity_matrix(
            full_vocab=full_vocab,
            repr_vocab=repr_vocab,
            processes=processes,
            matrix=sim_alignment_matrix,
            tqdm_desc=f'{EMOJI} Computing similarity matrix between '
                      f'full/repr vocab with global alignment ({sim_alignment_matrix})'
        )
    else:
        alphabet = set(itt.chain.from_iterable(repr_vocab))
        alphabet.add(" ")

        ngram_to_index = {
            ngram: i
            for i, ngram in enumerate(["".join(t) for t in itt.product(alphabet, repeat=n)])
        }

        if sim == "ngram_intersec":
            secho(f'Computing n-gram sparse similarities with {sim}')
            repr_similarity_matrix = compute_similarity_matrix_ngram_sparse(
                full_vocab=repr_vocab,
                repr_vocab=repr_vocab,
                ngram_to_index=ngram_to_index,
                n=n,
            )
            full_similarity_matrix = compute_similarity_matrix_ngram_sparse(
                full_vocab=full_vocab,
                repr_vocab=repr_vocab,
                ngram_to_index=ngram_to_index,
                n=n,
            )
        else:  # sim == 'ngram_sim'
            secho(f'Computing n-gram similarities with {sim}')
            repr_similarity_matrix = compute_similarity_matrix_ngram_parallel(
                full_vocab=repr_vocab,
                repr_vocab=repr_vocab,
                n=n,
                ngram_to_index=ngram_to_index,
                processes=processes,  # Extra because this gets multi-processed
            )
            full_similarity_matrix = compute_similarity_matrix_ngram_parallel(
                full_vocab=full_vocab,
                repr_vocab=repr_vocab,
                n=n,
                ngram_to_index=ngram_to_index,
                processes=processes,  # Extra because this gets multi-processed
            )

    repr_similarity_matrix_path = os.path.join(output, f"repr_similarity_matrix.npy")
    secho(f"Saving the repr similarity matrix for the full vocabulary to {repr_similarity_matrix_path}")
    np.save(repr_similarity_matrix_path, repr_similarity_matrix, allow_pickle=False)

    full_similarity_matrix_path = os.path.join(output, f"full_similarity_matrix.npy")
    secho(f"Saving the full similarity matrix for the full vocabulary to {full_similarity_matrix_path}")
    np.save(full_similarity_matrix_path, full_similarity_matrix, allow_pickle=False)

    optim_folder = os.path.join(output, 'optim')
    os.makedirs(optim_folder, exist_ok=True)

    if n_components is None:
        n_components = int(0.5 + len(repr_vocab) * 2 / 3)

    optimize_projections(
        output=optim_folder,
        repr_similarity_matrix=repr_similarity_matrix,
        full_similarity_matrix=full_similarity_matrix,
        n_components=n_components,
        similarity_type=sim,
        use_gpu=use_gpu,
    )

    if use_gpu:  # only shut down after all loops have used this function
        import cudamat as cm
        cm.shutdown()

    secho(f"done. Enjoy your {make_ratvec(3)}")


@click.command()
@click.option('-f', '--full-sim-matrix-file', required=True,
              help="A path to a file containing the full vocabulary similarity matrix")
@click.option('-r', '--repr-sim-matrix-file',
              help="A path to a file containing the repr similarity matrix")
@click.option('-d', '--output', type=click.Path(file_okay=False, dir_okay=True), required=True,
              help="Output folder for the KPCA embeddings")
@click.option('--n-components', type=int, show_default=True,
              help="Number of principal components of the embeddings. "
                   "If not specified, defaults to 2/3 the size of the representative vocabulary")
@click.option('--sim', type=click.Choice(ALLOWED_SIMILARITIES),
              default="ngram_intersec", show_default=True,
              help="Similarity function: 'ngram_sim' (n-gram similarity),'sorensen_plus' "
                   "(Sørensen–Dice index for n-grams)")
@click.option('--use-gpu', is_flag=True)
def infer(
        full_sim_matrix_file: str,
        repr_sim_matrix_file: str,
        output: str,
        n_components: int,
        sim: str,
        use_gpu: bool,
):
    """Load pre-computed similarity matrix."""
    secho(f"Loading the repr similarity matrix for the full vocabulary to {repr_sim_matrix_file}")
    repr_similarity_matrix = np.load(repr_sim_matrix_file)

    secho(f"Loading the full similarity matrix for the full vocabulary to {full_sim_matrix_file}")
    full_similarity_matrix = np.load(full_sim_matrix_file)

    optim_folder = os.path.join(output, 'optim')
    os.makedirs(optim_folder, exist_ok=True)

    optimize_projections(
        output=optim_folder,
        repr_similarity_matrix=repr_similarity_matrix,
        full_similarity_matrix=full_similarity_matrix,
        n_components=n_components,
        similarity_type=sim,
        use_gpu=use_gpu,
    )

    if use_gpu:  # only shut down after all loops have used this function
        import cudamat as cm
        cm.shutdown()

    secho(f"done. Enjoy your {make_ratvec(3)}")


def optimize_projections(
        *,
        output: str,
        repr_similarity_matrix,
        full_similarity_matrix,
        n_components: int,
        similarity_type: str,
        use_gpu: bool,
) -> None:
    """

    :param output: The output folder
    :param repr_similarity_matrix: A square matrix with dimensions |repr| x |repr|
    :param full_similarity_matrix: A rectangular matrix with dimensions |full| x |repr|
    :param n_components:
    :return:
    """
    khc = (
        (kernel_name, KERNEL_TO_PROJECTION[kernel_name], hyperparam)
        for kernel_name, hyperparams in kernels.items()
        for hyperparam in hyperparams
    )

    for kernel_name, project_with_kernel, hyperparam in khc:
        # Make output folder for the optimization with this kernel/hyper-parameter pair
        param_folder = os.path.join(output, f'{kernel_name}_{hyperparam}')
        os.makedirs(param_folder, exist_ok=True)

        secho(f"({kernel_name}/{hyperparam}) calculating normalized/symmetric kernel matrix")
        repr_kernel_matrix = project_with_kernel(repr_similarity_matrix, hyperparam)
        repr_kernel_matrix_normalized = normalize_kernel_matrix(repr_kernel_matrix)

        secho(f"({kernel_name}/{hyperparam}) solving eigenvector/eigenvalues problem")
        eigenvalues, eigenvectors = eigh(repr_kernel_matrix_normalized)

        # Calculate alphas
        repr_alphas = np.column_stack([
            eigenvectors[:, -i]
            for i in range(1, n_components + 1)
        ])
        # Save Alphas
        _alphas_path = os.path.join(param_folder, f"alphas.p")
        secho(f"({kernel_name}/{hyperparam}) outputting alphas to {_alphas_path}")
        with open(_alphas_path, "wb") as file:
            pickle.dump(repr_alphas, file)

        # Calculate lambdas
        repr_lambdas = [
            eigenvalues[-i]
            for i in range(1, n_components + 1)
        ]
        # Save lambdas
        _lambdas_path = os.path.join(param_folder, f"lambdas.p")
        secho(f"({kernel_name}/{hyperparam}) outputting lambdas to {_lambdas_path}")
        with open(_lambdas_path, 'wb') as file:
            pickle.dump(repr_lambdas, file)

        secho(f"({kernel_name}/{hyperparam}) projecting known vocabulary to KPCA embeddings")
        repr_projection_matrix = repr_alphas / repr_lambdas

        # Calculate KPCA matrix
        if similarity_type == "ngram_intersec":  # There is no additional kernel function on top of the similarity function
            kpca_matrix = project_full_vocab_linear(
                projection_matrix=repr_projection_matrix,
                similarity_matrix=full_similarity_matrix,
            )
        elif use_gpu:
            kpca_matrix = project_words_gpu(
                projection_matrix=repr_projection_matrix,
                similarity_matrix=full_similarity_matrix,
                kernel_name=kernel_name,
                hyperparam=hyperparam,
            )
        else:
            kpca_matrix = project_similarity_matrix(
                projection_matrix=repr_projection_matrix,
                similarity_matrix=full_similarity_matrix,
                kernel_name=kernel_name,
                hyperparam=hyperparam,
            )

        # Save KPCA matrix
        _kpca_path = os.path.join(param_folder, f"kpca.npy")
        secho(f"({kernel_name}/{hyperparam}) outputting KPCA matrix to {_kpca_path}")
        np.save(_kpca_path, kpca_matrix)


def project_full_vocab_linear(projection_matrix, similarity_matrix):
    return similarity_matrix.dot(projection_matrix)


def compute_splits(
        elements: List[Any],
        split_size,
        processes: int,
        use_tqdm: bool = True,
) -> List[List[Any]]:
    it = range(processes)
    if use_tqdm:
        it = tqdm(it, desc=f"{EMOJI} computing splits")
    return [
        elements[index * split_size: (index + 1) * split_size]
        for index in it
    ]


def get_ngram_elements(*, repr_vocab, full_vocab, n: int, ngram_to_index):
    ngram_repr_vocab = _get_ngram_elements_helper(
        repr_vocab,
        ngram_to_index=ngram_to_index,
        n=n,
        desc=f"{EMOJI} Computing n-grams for representative vocabulary",
    )
    ngram_vocab = _get_ngram_elements_helper(
        full_vocab,
        ngram_to_index=ngram_to_index,
        n=n,
        desc=f"{EMOJI} Computing n-grams for full vocabulary",
    )
    return list(itt.product(ngram_vocab, ngram_repr_vocab))


def _get_ngram_elements_helper(strings, *, ngram_to_index, n: int, desc=None):
    return [
        [
            ngram_to_index[t]
            for t in ngrams(string, n)
        ]
        for string in tqdm(strings, desc=desc)
    ]


def compute_similarity_matrix_ngram_parallel(
        *,
        repr_vocab,
        full_vocab,
        processes,
        n,
        ngram_to_index,
) -> np.ndarray:
    """

    :param repr_vocab:
    :param full_vocab:
    :param processes:
    :param n:
    :param ngram_to_index:
    :return:
    """
    from ratvec.similarity import n_gram_sim_list

    secho(f"Splitting data for computing similarities in {processes} processes")
    elements = get_ngram_elements(
        full_vocab=full_vocab,
        repr_vocab=repr_vocab,
        ngram_to_index=ngram_to_index,
        n=n,
    )
    compute_similarities_on_splits = partial(n_gram_sim_list, n_ngram=n)
    return _calculate_similarity_matrix_parallel(
        full_vocab=full_vocab,
        repr_vocab=repr_vocab,
        processes=processes,
        elements=elements,
        compute_similarities_on_splits=compute_similarities_on_splits,
    )


def _calculate_similarity_matrix_parallel(
        *,
        full_vocab,
        repr_vocab,
        processes,
        elements,
        compute_similarities_on_splits: Callable,
) -> np.ndarray:
    full_vocab_len = len(full_vocab)
    repr_vocab_len = len(repr_vocab)
    split_size = ceil((full_vocab_len * repr_vocab_len) / processes)
    splits: List[List[Any]] = compute_splits(
        elements=elements,
        split_size=split_size,
        processes=processes,
    )

    secho(f'Computing similarities in {processes} processes')
    with multiprocessing.Pool(processes=processes) as pool:
        res = pool.map(compute_similarities_on_splits, splits)
        res = np.hstack(res)

    return res.reshape(full_vocab_len, repr_vocab_len)


def compute_similarity_matrix_ngram_sparse(
        *,
        repr_vocab,
        full_vocab,
        ngram_to_index,
        n: int,
        use_tqdm: bool = True,
):
    v_size = len(ngram_to_index)

    # R is the transposed matrix of the one-hot representations of the representative vocab
    R = dok_matrix((v_size, len(repr_vocab)), dtype=int)
    R_ng = np.zeros([len(repr_vocab)], dtype=int)

    it_1 = np.arange(R.shape[1])
    if use_tqdm:
        it_1 = tqdm(it_1, desc=f"{EMOJI} compute one hot representation of representative vocabulary")
    for i in it_1:
        ng = ngrams(repr_vocab[i], int(n))
        R_ng[i] = len(ng)
        for j in range(len(ng)):
            R[ngram_to_index[ng[j]], i] = 1

    R.tocsr()

    V = dok_matrix((len(full_vocab), v_size), dtype=int)
    V_ng = np.zeros([len(full_vocab)], dtype=int)

    it_2 = np.arange(V.shape[0])
    if use_tqdm:
        it_2 = tqdm(it_2, desc=f"{EMOJI} compute one hot representation of full vocabulary")
    for i in it_2:
        ng = ngrams(full_vocab[i], int(n))
        V_ng[i] = len(ng)
        for j in range(len(ng)):
            try:
                V[i, ngram_to_index[ng[j]]] = 1
            except KeyError:
                pass
    V.tocsr()

    L = np.empty([len(V_ng), len(R_ng)], dtype=int)

    it_3 = range(len(V_ng))
    if use_tqdm:
        it_3 = tqdm(it_3, desc=f"{EMOJI} Compute normalization matrix with maximum number of n-grams for the proteins")
    for i in it_3:
        for j in range(len(R_ng)):
            L[i, j] = max(V_ng[i], R_ng[j])

    return V.dot(R).toarray() / L


def calculate_global_alignment_similarity_matrix(
        *,
        full_vocab,
        repr_vocab,
        matrix: str,
        processes: int,
        use_tqdm: bool = True,
        tqdm_desc=None,
):
    from ratvec.similarity_slow import global_alignment_similarity

    func = partial(global_alignment_similarity, matrix=matrix)
    iterable = itt.product(full_vocab, repr_vocab)
    if use_tqdm:
        iterable = tqdm(
            iterable,
            total=len(full_vocab) * len(repr_vocab),
            desc=tqdm_desc,
        )

    with multiprocessing.Pool(processes=processes) as pool:
        res = pool.starmap(
            func=func,
            iterable=iterable,
        )
        res = np.array(res)
    return res.reshape(len(full_vocab), len(repr_vocab))


def project_similarity_matrix(projection_matrix, similarity_matrix, kernel_name, hyperparam):
    """
    :param projection_matrix: Obtained from the eigen-decomposition
    :param similarity_matrix: Matrix with the evaluation of the similarity function on each word-repr. word pair
    :param kernel_name: Kernel function to be composed with the similarity function for projection
    :param hyperparam: Hyper-parameter of the additional kernel function
    
    :return: matrix of same shape is sim_matrix
    """
    if kernel_name == "poly":  # Polynomial kernel
        kernel = similarity_matrix ** hyperparam
    elif kernel_name == "rbf":  # RBF kernel
        kernel = np.exp(-hyperparam * np.square((1 - similarity_matrix)))
    else:  # linear
        kernel = similarity_matrix + hyperparam

    return kernel.dot(projection_matrix)


def project_words_gpu(projection_matrix, similarity_matrix, kernel_name, hyperparam):
    import cudamat as cm
    if kernel_name == "poly":
        k = cm.pow(cm.CUDAMatrix(similarity_matrix), hyperparam)
    elif kernel_name == 'rbf':
        k = cm.exp((cm.pow(cm.CUDAMatrix(1 - similarity_matrix), 2)).mult(-hyperparam))
    else:
        raise NotImplementedError(f'{kernel_name} not yet implemented for GPU')

    return cm.dot(k, cm.CUDAMatrix(projection_matrix)).asarray()


if __name__ == '__main__':
    main()
