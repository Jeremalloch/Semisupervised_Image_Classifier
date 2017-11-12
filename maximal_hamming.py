import math
import numpy as np
import itertools
import random


def hamming(str1, str2):
    return sum(x != y for x, y in zip(str1,str2))

def gen_max_hamming_set(N):
    """
    Generates maximal hamming distance permutation set
    N - the number of permutations in the returned set
    """
    full_permutation_set = itertools.permutations(range(9),9)
    max_hamming_sets = []
    j = random.randint(0,math.factorial(9))
    D = []
    D_ = []
    for i in range(N):
        max_hamming_sets.append(full_permutation_set[j])
        del full_permutation_set[j]
        D = [[hamming(P1,P2) for P1 in max_hamming_sets] for P2 in full_permutation_set]
        D_ = list(map(sum,D))
        j = max(D_)

    return max_hamming_sets

#  def gen_max_hamming_set(N):
#      """
#      Generates maximal hamming distance permutation set
#      N - the number of permutations in the returned set
#      """
#      full_permutation_set = np.negative(np.ones([math.factorial(9),9],dtype=int)
#      max_hamming_sets = np.negative(np.ones([N,9], dtype=int))
#      for i, el in enumerate(itertools.permutations(range(9),9)):
#          full_permutation_set[i] = el
#      j = random.randint(0,math.factorial(9))
#      D = np.zeros(dtype=int)
#      D_ = np.zeros(dtype=int)
#
#      for i in range(N):
#          max_hamming_sets[i] = full_permutation_set[j]
#          full_permutation_set[j] = np.negative(np.ones(9,dtype=int))
#          D = [[hamming(P1,P2) for P1 in full_permutation_set] for P2 in max_hamming_sets]
#          hamming()
#          D_ = np.sum(D, axis=0)
#          j = max(D_)
#
#      return max_hamming_sets
