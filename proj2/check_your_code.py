import copy
import math

import numpy as np
from sklearn.datasets import load_iris

from knn import dist, normalize_data


def _check_dist(data):
    # Distance with itself should be zero
    self_dist = dist(data[0], data[0])
    if not np.allclose(self_dist, np.zeros_like(self_dist), atol=1E-5):
        raise ValueError("Distance for vector to itself not close")

    # Simple test case
    simple_vec = np.array([1., -2., 3., -4.])
    var_dist = dist(simple_vec, np.zeros_like(simple_vec))
    if not np.allclose(var_dist, np.array([30])):
        raise ValueError("Distance function appears to be wrong")

    first_dist = dist(data[0], data[1])
    expected = 0.29
    expected_squared = math.sqrt(expected)
    if abs(first_dist - expected) > 1E-4:
        if abs(first_dist - expected_squared) < 1E-4:
            raise ValueError("You are taking the square root in the distance.  Don't do that")
        raise ValueError("Distance function on real data appears to be wrong")


def _check_rangenorm():
    ones = np.ones([3, 3])

    train, test = normalize_data(ones, ones, rangenorm=True, varnorm=False, exnorm=False)
    for nan_check, name in ((train, "train"), (test, "test")):
        if not np.allclose(nan_check, np.zeros_like(nan_check), equal_nan=True):
            raise ValueError(f"NaNs are not properly set to zero for {name}.  You should do that")

    # Check the range is correct
    rand_data = np.random.rand(5, 3)
    rand_data, _ = normalize_data(rand_data, rand_data, rangenorm=True, varnorm=False, exnorm=False)
    for i in range(rand_data.shape[1]):
        col_vec = rand_data[:, i]
        if abs(np.min(col_vec) + 1.) > 1E-5:
            raise ValueError("Minimum does not appear to set to -1 as expected")
        if abs(np.max(col_vec) - 1.) > 1E-5:
            raise ValueError("Maximum does not appear to set to 1 as expected")
    # Check the normalization
    data = np.array([[ 1.0, 10.,  3.,  4.],
                     [-1.0, 20.,  4.,  1.],
                     [ 0.5, 40.,  7.,  0.],
                     [-0.5, 50.,  6.,  3.],
                     [ 0.,  30.,  5.,  2.]])
    copy_data = copy.deepcopy(data)
    x1, x2 = normalize_data(data, data, rangenorm=True, varnorm=False, exnorm=False)
    if not np.allclose(x1, x2):
        raise ValueError("Test/train not identical for identical data")
    train, test = normalize_data(data, 10 * data + 5, rangenorm=True, varnorm=False, exnorm=False)
    if not np.max(test) > 20:
        raise ValueError("You seem to be normalizing based on test data")

    if train.shape != data.shape or test.shape != data.shape:
        return ValueError("Training or test set size mismatch")
    if not np.alltrue(copy_data == data):
        raise ValueError("It appears you are modifying the input data")

    col_sum = np.sum(train, axis=1)
    if np.allclose(col_sum, np.zeros_like(col_sum)):
        raise ValueError("Normalization not right")


def _check_varnorm(data):
    x = np.zeros([5, 3])
    train, test = normalize_data(x, x, rangenorm=False, varnorm=True, exnorm=False)
    # Verify you handle NaNs
    for arr, name in ((train, "Train"), (test, "Test")):
        if not np.allclose(arr, np.zeros_like(arr), equal_nan=True):
            raise ValueError(f"{name} is not all zeros.  You probably have NaNs")

    train, test = normalize_data(data, data, rangenorm=False, varnorm=True, exnorm=False)
    for arr, name in ((train, "Train"), (test, "Test")):
        tot = np.sum(np.abs(arr))
        if abs(tot - 504.434773) > 1E-3:
            raise ValueError(f"Check your math.  {name} does not seem to aggregate")

    eye = np.eye(2)
    train, test = normalize_data(eye, eye, rangenorm=False, varnorm=True, exnorm=False)
    expect = np.array([[1, -1], [-1, 1]])
    for arr, name in ((train, "Train"), (test, "Test")):
        if not np.allclose(expect, arr):
            raise ValueError(f"Check your simple math.  {name} does not seem to aggregate")


def _check_exnorm():
    r""" Checks var norm """
    x = np.zeros([5, 3])
    train, test = normalize_data(x, x, rangenorm=False, varnorm=False, exnorm=True)
    if not np.allclose(train, test, equal_nan=True):
        raise ValueError("Train/test do not match")

    x_arr = [[ 1, 2, 3, 4],
             [-1, 0, 1, 2]]
    x = np.array(x_arr)
    y = np.array([[1., 1., 1., 1.]])
    train, test = normalize_data(x, y, rangenorm=False, varnorm=False, exnorm=True)

    norms = [math.sqrt(sum(float(val) ** 2 for val in arr)) for arr in x_arr]
    x_arr_norm = [[x / norm for x in arr] for arr, norm in zip(x_arr, norms)]
    if not np.allclose(train, np.array(x_arr_norm)):
        raise ValueError("Train vector seems wrong")
    if not np.allclose(test, np.full_like(test, fill_value=0.5)):
        raise ValueError("Test vector seems wrong")


def _main():
    iris = load_iris()
    data, target = iris['data'], iris['target']

    _check_dist(data)
    _check_rangenorm()
    _check_varnorm(data)
    _check_exnorm()


if __name__ == '__main__':
    _main()