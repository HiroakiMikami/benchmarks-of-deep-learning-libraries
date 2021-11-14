import numpy
cimport numpy


def length_of_longest_common_subsequence(str0: str, str1: str) -> int:
    N = len(str0)
    M = len(str1)
    table = numpy.zeros(shape=(N + 1, M + 1), dtype=numpy.int64)

    for i in range(N):
        for j in range(M):
            if str0[i] == str1[j]:
                table[i + 1, j + 1] = table[i, j] + 1
            else:
                table[i + 1, j + 1] = max(table[i + 1, j], table[i, j + 1])
    return int(table[-1, -1].item())
