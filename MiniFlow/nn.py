#!/usr/local/bin/python
# -*- coding: UTF-8 -*-
"""
    nn
    ~~

    Run the MiniFlow neural network
"""

# Define 2 `Input` nodes.
x, y = Input(), Input()

# Define an `Add` node, the two above `Input` nodes being the input.
add = Add(x, y)

# The value of `x` and `y` will be set to 10 and 20 respectively.
feed_dict = {x: 10, y: 20}

# Sort the nodes with topological sort.
sorted_nodes = topological_sort(feed_dict=feed_dict)
output = forward_pass(f, sorted_nodes)

# NOTE: because topological_sort set the values for the `Input` nodes we could also access
# the value for x with x.value (same goes for y).
print("{} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output))