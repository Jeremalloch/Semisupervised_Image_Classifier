import math
import itertools
import random
import time


def hamming_distance(str1, str2):
    """
    Calculates the hamming distance between two strings
    """
    return sum(x != y for x, y in zip(str1,str2))

def gen_max_hamming_set(N):
    """
    Generates maximal hamming distance permutation set
    N - the number of permutations in the returned set
    """
    full_permutation_set = list(itertools.permutations(range(9),9))
    max_hamming_sets = []
    j = random.randint(0,math.factorial(9))
    D = []
    D_ = []
    for i in range(N):
        max_hamming_sets.append(full_permutation_set[j])
        del full_permutation_set[j]
        D = [[hamming_distance(P1,P2) for P1 in max_hamming_sets] for P2 in full_permutation_set]
        D_ = list(map(sum,D))
        j = max(D_)

    return max_hamming_sets

num_permutations = 25
a = time.time()
B = gen_max_hamming_set(num_permutations)
b = time.time()

print("Took {} seconds to generate {} permutations".format(b-a,num_permutations))
