import numpy as np

a = ["a", "b", "c", "d", "e", "f", "g"]
b = np.random.randint(0, 50, size=7)
forest = dict(zip(a, b))
weight = list(forest.values())
weight.sort()
for value in weight:

print(forest)
print(weight)
