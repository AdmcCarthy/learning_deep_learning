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

from miniflow import (
                      Input,
                      Add,
                      topological_sort,
                      forward_pass,
                      Linear
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

    print(output) # should be 12.7 with this example

if __name__ == "__main__":
    addition_2()