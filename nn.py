import numpy as np

def element_wise_addition(a, b):
    assert len(a) == len(b), "Both lists must be of the same length."

    for i in range(len(a)):
        a[i] = a[i] + b[i]
    return a

def element_wise_multiplication(a, b):
    assert len(a) == len(b), "Both lists must be of the same length."

    c = [0 for _ in range(len(a))]

    for i in range(len(a)):
        c[i] = a[i] * b[i]
    return c

def vector_sum(v):
    total = 0
    for i in range(len(v)):
        total += v[i]
    return total

def vector_average(v):
    total = vector_sum(v)
    return total / len(v)

def weighted_sum(x, w):
    pred = element_wise_multiplication(x, w)
    return vector_sum(pred)

def neural_network(inputs, weights):
    pred = weighted_sum(inputs, weights)
    return pred

# Some sort of crude way for doins some probablistic calculations. Not clear yet.
a = [0,1,0,1]
b = [1,0, 1, 0]
c = [0,1,1,0]
d = [0.5,0,0.5,0]
e = [0,1,-1,0]

assert weighted_sum(a, b) == 0
assert weighted_sum(b, c) == 1
assert weighted_sum(b, d) == 1
assert weighted_sum(c, c) == 2
assert weighted_sum(d, d) == 0.5
assert weighted_sum(c, e) == 0
assert weighted_sum(e, e) == 2

number_of_toes = [8.5, 9.5, 10.0, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

print(neural_network([number_of_toes[0], wlrec[0], nfans[0]], [0.1, 0.2, 0]))