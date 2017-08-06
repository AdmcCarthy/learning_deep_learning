#!/usr/local/bin/python
# -*- coding: UTF-8 -*-
"""
    neural net
    ~~~~~~~~~~

    Learn about neural networks.

    Change the function(s) in
    if __name__ to run different
    examples.
"""

import numpy as np
import pandas as pd


def and_ex():
    """
    Demonstrate an AND perceptron
    by just changing the weights and bias
    """
    # TODO: Set weight1, weight2, and bias
    weight1 = 0.5
    weight2 = 0.5
    bias = -0.7

    # DON'T CHANGE ANYTHING BELOW
    # Inputs and outputs
    test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    correct_outputs = [False, False, False, True]
    outputs = []

    # Generate and check output
    for test_input, correct_output in zip(test_inputs, correct_outputs):
        linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
        output = int(linear_combination >= 0)
        is_correct_string = 'Yes' if output == correct_output else 'No'
        outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

    # Print output
    num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
    output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
    if not num_wrong:
        print('Nice!  You got it all correct.\n')
    else:
        print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
    print(output_frame.to_string(index=False))


def not_ex():
    """
    Demonstrate a NOT perceptron
    by just changing the weights and bias
    """
    # TODO: Set weight1, weight2, and bias
    weight1 = 3
    weight2 = -5
    bias = 1

    # DON'T CHANGE ANYTHING BELOW
    # Inputs and outputs
    test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    correct_outputs = [True, False, True, False]
    outputs = []

    # Generate and check output
    for test_input, correct_output in zip(test_inputs, correct_outputs):
        linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
        output = int(linear_combination >= 0)
        is_correct_string = 'Yes' if output == correct_output else 'No'
        outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

    # Print output
    num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
    output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
    if not num_wrong:
        print('Nice!  You got it all correct.\n')
    else:
        print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
    print(output_frame.to_string(index=False))


def simplest_nn():
    """
    Make a very simple neural network
    """
    def sigmoid(x):
        # Implement sigmoid function
        return 1/(1 + np.exp(-x))

    inputs = np.array([0.7, -0.3])
    weights = np.array([0.1, 0.8])
    bias = -0.1

    # Calculate the output
    output = sigmoid(np.dot(weights, inputs) + bias)

    print('Output:')
    print(output)


def gradient_descent():
    def sigmoid(x):
        """
        Calculate sigmoid
        """
        return 1/(1+np.exp(-x))

    def sigmoid_prime(x):
        """
        # Derivative of the sigmoid function
        """
        return sigmoid(x) * (1 - sigmoid(x))

    learnrate = 0.5
    x = np.array([1, 2, 3, 4])
    y = np.array(0.5)

    # Initial weights
    w = np.array([0.5, -0.5, 0.3, 0.1])

    # Calculate one gradient descent step for each weight
    # Note: Some steps have been consilated, so there are
    #       fewer variable names than in the above sample code

    # Calculate the node's linear combination of inputs and weights
    h = np.dot(x, w)

    # Calculate output of neural network
    nn_output = sigmoid(h)

    # Calculate error of neural network
    error = y - nn_output

    # TCalculate the error term
    # Remember, this requires the output gradient, which we haven't
    # specifically added a variable for.
    error_term = error * sigmoid_prime(h)
    # Note: The sigmoid_prime function calculates sigmoid(h) twice,
    #       but you've already calculated it once. You can make this
    #       code more efficient by calculating the derivative directly
    #       rather than calling sigmoid_prime, like this:
    # error_term = error * nn_output * (1 - nn_output)

    # Calculate change in weights
    del_w = learnrate * error_term * x

    print('Neural Network output:')
    print(nn_output)
    print('Amount of Error:')
    print(error)
    print('Change in Weights:')
    print(del_w)


def forward_pass():
    """
    Implement a forward pass
    """
    def sigmoid(x):
        """
        Calculate sigmoid
        """
        return 1/(1+np.exp(-x))

    # Network size
    N_input = 4
    N_hidden = 3
    N_output = 2

    np.random.seed(42)
    # Make some fake data
    X = np.random.randn(4)

    weights_input_to_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
    weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))

    # Make a forward pass through the network
    hidden_layer_in = np.dot(X, weights_input_to_hidden)
    hidden_layer_out = sigmoid(hidden_layer_in)

    print('Hidden-layer Output:')
    print(hidden_layer_out)

    output_layer_in = np.dot(hidden_layer_out, weights_hidden_to_output)
    output_layer_out = sigmoid(output_layer_in)

    print('Output-layer Output:')
    print(output_layer_out)


def backpropagation():
    """
    Implement backproagation.
    """
    def sigmoid(x):
        """
        Calculate sigmoid
        """
        return 1 / (1 + np.exp(-x))

    x = np.array([0.5, 0.1, -0.2])
    target = 0.6
    learnrate = 0.5

    weights_input_hidden = np.array([[0.5, -0.6],
                                    [0.1, -0.2],
                                    [0.1, 0.7]])

    weights_hidden_output = np.array([0.1, -0.3])

    # Forward pass
    hidden_layer_input = np.dot(x, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
    output = sigmoid(output_layer_in)

    # Backwards pass
    # Calculate output error
    error = target - output

    # Calculate error term for output layer
    output_error_term = error * output * (1 - output)

    # Calculate error term for hidden layer
    hidden_error_term = np.dot(output_error_term, weights_hidden_output) * \
                               hidden_layer_output * (1 - hidden_layer_output)

    # Calculate change in weights for hidden layer to output layer
    delta_w_h_o = learnrate * output_error_term * hidden_layer_output

    # Calculate change in weights for input layer to hidden layer
    delta_w_i_h = learnrate * hidden_error_term * x[:, None]

    print('Change in weights for hidden layer to output layer:')
    print(delta_w_h_o)
    print('Change in weights for input layer to hidden layer:')
    print(delta_w_i_h)


if __name__ == "__main__":
    and_ex()
    not_ex()
    simplest_nn()
    gradient_descent()
    forward_pass()
    backpropagation()
