from cpu.longest_common_subsequence import length_of_longest_common_subsequence


def test_lcs() -> None:
    n = length_of_longest_common_subsequence(
        "agcat", "gac"
    )
    assert n == 2
