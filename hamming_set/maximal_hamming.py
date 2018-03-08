import itertools
import random
import sys
import time
from time import strftime, localtime
import numpy as np
from numba import vectorize, njit, prange, jit, u1


@jit(u1(u1[:], u1[:]))
def hamming_distance(array1, array2):
    """
    Calculates the hamming distance between two numpy arrays of uint8s
    """
    if (array1.shape != array1.shape):
        raise ValueError("Shape of array1 & array2 does not match")
    distance = 0
    for i in range(array1.shape[0]):
        if (array1[i] != array2[i]):
            distance += 1
    return distance


@jit
def gen_max_hamming_set(N):
    """
    Generates maximal hamming distance permutation set
    N - the number of permutations in the returned set
    """
    # There a 9! permutations, factorial function avoided for numba
    NUM_PERMUTATIONS = 362880
    # Use a list here since we'll be popping elements from random locations,
    # and having to delete those elements, so no speedup using a numpy array
    #  full_permutation_set = list(itertools.permutations(range(9),9))
    full_permutation_set = np.zeros((NUM_PERMUTATIONS, 9), dtype=np.uint8)
    for i, el in enumerate(itertools.permutations(range(9), 9)):
        full_permutation_set[i] = el
    max_hamming_sets = np.zeros((N, 9), dtype=np.uint8)
    j = random.randint(0, NUM_PERMUTATIONS)
    D = np.zeros((N, NUM_PERMUTATIONS), dtype=np.uint8)
    #  pdb.set_trace()
    for i in range(N):
        a = time.time()
        max_hamming_sets[i] = full_permutation_set[j]

        # Delete the jth element by shifting all elements to the right of it
        # one left, overwriting the jth element
        for index in range(j, NUM_PERMUTATIONS - (i + 1)):
            full_permutation_set[index] = full_permutation_set[index + 1]
        full_permutation_set[NUM_PERMUTATIONS -
                             (i + 1)] = np.zeros((1, 9), dtype=np.uint8)

        # Loop implementation of hamming distance calculation
        a1 = time.time()
        for j in range(i + 1):
            for k in range(NUM_PERMUTATIONS - i):
                D[j, k] = hamming_distance(
                    max_hamming_sets[j], full_permutation_set[k])
        b1 = time.time()
        print("Took {} seconds to calculate hamming distances".format(b1 - a1))
        #  j = np.max(np.sum(D, axis=0))
        # Should be the index j was found at
        j = np.argmax(np.sum(D, axis=0))

        # Since the dimension of D are ix(NUM_PERMUTATIONS-i), zero out the NUM_PERMUTATIONS-i column
        # since its no longer part of the matrix, shouldn't be considered when
        # determining j
        D[:, NUM_PERMUTATIONS - (i + 1)] = np.zeros((N), dtype=np.uint8)

        b = time.time()
        print("Took {} seconds to run loop once".format(b - a))

    return max_hamming_sets


def main(num_permutations=25):
    """
    Main function to generate hamming_distance set
    """

    start_time = time.time()
    permutations = gen_max_hamming_set(num_permutations)
    end_time = time.time()
    print("Took {} seconds to generate {} permutations".format(
        end_time - start_time, num_permutations))

    print(permutations)

    output_name = "maxHammingSet_of_{}_on_{}.txt".format(
        num_permutations, strftime("%b_%d_%H:%M:%S", localtime()))

    with open(output_name, 'w') as f:
        for row in permutations:
            for number in row:
                f.write("{} ".format(number))
            f.write("\n")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(int(sys.argv[1]))
    else:
        print("Number of permutations not specified, using 25")
        main()
