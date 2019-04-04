#   Copyright 2018 Fraunhofer IAIS
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
cimport numpy as np
cimport cython
from functools import partial
import itertools as itt
from tqdm import tqdm
from typing import List, Tuple


def cosine(v,M):
	return 1- M.dot(v) / (np.sqrt(np.dot(v,v))*np.sqrt(np.einsum('ij,ij->i', M, M)))

def euclidean(v,M):   
	return np.sqrt(np.sum((M-v)*(M-v), axis=1))

cpdef ngrams(s, int n):
	"""List of ngrams of the string after padding with a filler (repetitions included). 
	
	Order of ngrams is kept .
	"""
	cdef int i
	s2 = " "+s+" "        
	return [s2[i:i+n] for i in range(len(s2)-n+1)]

cpdef double ngram_sim(ng_a:List[int], ng_b:List[int], int n= 2) except? -1:
	"""Binary version of n-gram similarity."""
	cdef int x_len,y_len,i,j, comp_res
	cdef int a,b #They actually represent ngrams indirectly!

	#ng_a = ngrams(x,n)
	#ng_b = ngrams(y,n)
	x_len = len(ng_a)
	y_len = len(ng_b)

	np_mem = np.zeros([x_len + 1, y_len + 1], dtype=np.intc)
	cdef int [:,:] mem_table = np_mem

	for i in range(1, x_len + 1):
		for j in range(1, y_len + 1):
			a = ng_a[i - 1]
			b = ng_b[j - 1]
			comp_res = a == b
			mem_table[i][j] = max(mem_table[i][j - 1], mem_table[i - 1][j], mem_table[i - 1][j - 1] + ( comp_res ) )
	
	return float(mem_table[x_len][y_len]) / float(max(x_len, y_len))


def n_gram_sim_list(tuples:List[Tuple[List[int], List[int]]], n_ngram:int):
	"""Computes the binary version of n-gram similarity of a list of tuples	
	"""
	#return [ngram_sim(tup[0], tup[1], n_ngram) for tup in tuples]
	
	#f = partial(ngram_sim, n_ngram)
	#return [
	#    f(x, y)
	#    for x, y in tuples
	#]
	cdef int list_len,i	
	list_len = len(tuples)
	np_mem = np.empty([list_len], dtype=np.float)
	cdef double [:] res_list = np_mem
	f = partial(ngram_sim, n=n_ngram)
	for i in tqdm(range(list_len), desc='Computing n-gram similarity splits'):
		res_list[i] = f(tuples[i][0],tuples[i][1])
	return np_mem

	

cpdef projectWord(word, alphas_lambdas_div, maxoids , int dim, int degree=2):
	"""Infers a KPCA embedding of the given word after the model is trained."""
	cdef int i,n
	n = len(maxoids)	

	k_np = np.empty([n], dtype=np.float_)
	cdef double [:] k= k_np


	##These commented lines are "readable" but not optimized version of the code
	#pair_sim = np.array([ similarity_function(word,t) for t in maxoids])
	#k = pair_sim**degree
	#return k.dot(alphas_lambdas_div)

	if (degree == 2):
		for i in range(n):
			k[i] = ngram_sim(word,maxoids[i])
			k[i] *= k[i]
	else:
		for i in range(n):
			k[i] = ngram_sim(word,maxoids[i])
			k[i] = k[i]**degree
	
	return np.dot(k, alphas_lambdas_div, out = np.empty([dim], dtype=np.float_) )	

