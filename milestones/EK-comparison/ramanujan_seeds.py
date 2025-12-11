"""
Generate seeds from the first 50 values of the Ramanujan integer partition function.
The Ramanujan partition function p(n) gives the number of ways to partition n.

We use the first 10 integer partition values (not partition counts, but indices).
These will be used as seeds for reproducibility.
"""

def get_ramanujan_partition_seeds(num_seeds: int = 10) -> list:
    """
    Get the first num_seeds values from integer partition sequences.
    For reproducibility, we use: 1, 2, 3, 5, 7, 11, 13, 17, 19, 23
    These are based on partition theory sequences.
    
    More formally, these can be computed from partition sequences, but
    we'll use the standard partition sequence values.
    """
    # Standard partition function values: p(n) for n = 0, 1, 2, ...
    # For seeds, we use the actual partition indices which follow patterns
    # First 10 meaningful partition sequence values
    partition_values = [
        1,      # p(0) = 1
        1,      # p(1) = 1
        2,      # p(2) = 2
        3,      # p(3) = 3
        5,      # p(4) = 5
        7,      # p(5) = 7
        11,     # p(6) = 11
        15,     # p(7) = 15
        22,     # p(8) = 22
        30,     # p(9) = 30
    ]
    return partition_values[:num_seeds]

if __name__ == "__main__":
    seeds = get_ramanujan_partition_seeds(10)
    print("Ramanujan partition seeds (first 10):")
    for i, seed in enumerate(seeds):
        print(f"  Seed {i}: {seed}")
