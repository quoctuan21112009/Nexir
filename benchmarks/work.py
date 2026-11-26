def compute(n: int) -> int:
    x = 1
    for _ in range(n):
        x = (x * 1664525 + 1013904223) % 2147483647
    return x


if __name__ == "__main__":
    print(compute(20_000_000))
