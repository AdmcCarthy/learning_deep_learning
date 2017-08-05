#!/usr/local/bin/python
# -*- coding: UTF-8 -*-
"""
    nn
    ~~

    Run the MiniFlow neural network

    Change the function(s) in
    if __name__ to run different
    examples.
"""

import numpy as np
from miniflow import (
                      Input,
                      Add,
                      topological_sort,
                      forward_pass,
                      Linear,
                      Sigmoid,
                      MSE
                      )


def addition_2():
    # Define 2 `Input` nodes.
    x, y = Input(), Input()

    # Define an `Add` node, the two above `Input` nodes being the input.
    f = Add(x, y)

    # The value of `x` and `y` will be set to 10 and 20 respectively.
    feed_dict = {x: 10, y: 20}

    # Sort the nodes with topological sort.
    sorted_nodes = topological_sort(feed_dict=feed_dict)
    output = forward_pass(f, sorted_nodes)

    # NOTE: because topological_sort set the values for the `Input` nodes we could also access
    # the value for x with x.value (same goes for y).
    print("{} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output))


def addition_3():
    x, y, z = Input(), Input(), Input()

    f = Add(x, y, z)

    feed_dict = {x: 4, y: 5, z: 10}

    graph = topological_sort(feed_dict)
    output = forward_pass(f, graph)

    # should output 19
    print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))


def linear():
    inputs, weights, bias = Input(), Input(), Input()

    f = Linear(inputs, weights, bias)

    feed_dict = {
        inputs: [6, 14, 3],
        weights: [0.5, 0.25, 1.4],
        bias: 2
    }

    graph = topological_sort(feed_dict)
    output = forward_pass(f, graph)

    print(output)  # should be 12.7 with this example


def linear_matrix():
    X, W, b = Input(), Input(), Input()

    f = Linear(X, W, b)

    X_ = np.array([[-1., -2.], [-1, -2]])
    W_ = np.array([[2., -3], [2., -3]])
    b_ = np.array([-3., -5])

    feed_dict = {X: X_, W: W_, b: b_}

    graph = topological_sort(feed_dict)
    output = forward_pass(f, graph)

    """
    Output should be:
    [[-9., 4.],
    [-9., 4.]]
    """
    print(output)


def sigmoid_test():
    X, W, b = Input(), Input(), Input()

    f = Linear(X, W, b)
    g = Sigmoid(f)

    X_ = np.array([[-1., -2.], [-1, -2]])
    W_ = np.array([[2., -3], [2., -3]])
    b_ = np.array([-3., -5])

    feed_dict = {X: X_, W: W_, b: b_}

    graph = topological_sort(feed_dict)
    output = forward_pass(g, graph)

    """
    Output should be:
    [[  1.23394576e-04   9.82013790e-01]
    [  1.23394576e-04   9.82013790e-01]]
    """
    print(output)


def mse():
    y, a = Input(), Input()
    cost = MSE(y, a)

    y_ = np.array([1, 2, 3])
    a_ = np.array([4.5, 5, 10])

    feed_dict = {y: y_, a: a_}
    graph = topological_sort(feed_dict)
    # forward pass
    forward_pass(graph)

    """
    Expected output

    23.4166666667
    """
    print(cost.value)


if __name__ == "__main__":
    mse()
