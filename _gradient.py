from ._package import np


def _difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h


def _partial_defference_quotient(f, v, i, h):
    x = v[i]
    return (f(x + h) - f(x)) / h


def _setp(v, direction, step_size):
    return [v_i + direction_i * step_size for (v_i, direction_i) in zip(v, direction)]


def _sum_of_square_gradient(v):
    return [2 * v_i for v_i in v]


def _vector_substract(v, w):
    return [v_i - w_i for v_i, w_i in zip(v, w)]


def _dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def _sum_of_square(v):
    return _dot(v, v)


def _distance(v, w):
    return _sum_of_square(_vector_substract(v, w))

