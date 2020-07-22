def test(a):
    a += ["--"]
    print(a)

if __name__ == "__main__":
    a = ["12"]
    test(a)
    print(a)
