a = [1, 2, 3]
b = [4, 5, 6]
c = [a, b]
d = list(filter(lambda x: x[0] == 1, c))
if b in d:
    print True
else:
    print False