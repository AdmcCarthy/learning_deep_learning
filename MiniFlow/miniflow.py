#!/usr/local/bin/python
# -*- coding: UTF-8 -*-
"""
    MiniFlow
    ~~~~~~~~

    A neural network implementation
    done only using Numpy.

    Based on material in Udacity
    Deep Learning Nanodegree.
"""


class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Node(s) from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this Node passes values
        self.outbound_nodes = []
        # For each inbound Node here, 
        # add this Node as an outbound Node to _that_ Node.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
        # A calculated value
        self.value = None

    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_nodes` and
        store the result in self.value
        """
        raise NotImplemented

class Input(Node):
    def __init__(self):
        # An Input node has no inbound nodes,
        # so no need to pass anything to the Node instantiator.
        Node.__init__(self)
    
    # NOTE: Input node is the only node where the value
    # may be passed as an argument to forward().
    #
    # All other node implementations should get the value
    # may be passed as an argument to forward().
    #
    # All other node implementations should get the value
    # of the previous node from self.inbound_nodes
    #
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value is not None:
            self.value = value

class Add(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        """
        This
        """


# Define 2 `Input` nodes.
x, y = Input(), Input()

# Define an `Add` node, the two above `Input` nodes being the input.
add = Add(x, y)

# The value of `x` and `y` will be set to 10 and 20 respectively.
feed_dict = {x: 10, y: 20}

# Sort the nodes with topological sort.
sorted_nodes = topological_sort(feed_dict=feed_dict)