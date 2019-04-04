# -*- coding: utf-8 -*-

import logging

from ratvec.similarity_slow import global_alignment_similarity

__all__ = [
    'n_gram_sim_list',
    'ngram_sim',
    'global_alignment_similarity',
]

logger = logging.getLogger(__name__)

try:
    import pyximport

    pyximport.install()
    from ratvec.similarity_fast import n_gram_sim_list, ngram_sim
except ImportError:
    logger.info('falling back to pure python implementation of n_gram_sim and n_gram_sim_list')
    from ratvec.similarity_slow import n_gram_sim_list, ngram_sim
