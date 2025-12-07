from layers import *
from functions import *

def build_neural_net(features, outputs, hidden_length, activation_fn, activation_prime):
    net = NeuralNet(binary_cross_entropy, binary_cross_entropy_prime)


    net.add(LinearLayer(features, hidden_length), name = "Hidden Layer 1")
    net.add(ActivationLayer(activation_fn, activation_prime), "Activation Layer")

    net.add(LinearLayer(hidden_length, hidden_length), name = "Hidden Layer 3")
    net.add(ActivationLayer(activation_fn, activation_prime), "Activation Layer")

    net.add(LinearLayer(hidden_length, 2 * hidden_length), name = "Hidden Layer 3")
    net.add(ActivationLayer(activation_fn, activation_prime), "Activation Layer")

    net.add(LinearLayer(2 * hidden_length, hidden_length), name = "Hidden Layer 3")
    net.add(ActivationLayer(activation_fn, activation_prime), "Activation Layer")

    net.add(LinearLayer(hidden_length, int(hidden_length / 2)), name = "Hidden Layer 3")
    net.add(ActivationLayer(activation_fn, activation_prime), "Activation Layer")
    
    net.add(LinearLayer(int(hidden_length / 2), int(hidden_length / 2)), name = "Hidden Layer 4")
    net.add(ActivationLayer(activation_fn, activation_prime), "Activation Layer")

    net.add(LinearLayer(int(hidden_length / 2), int(hidden_length / 2)), name = "Hidden Layer 4")
    net.add(ActivationLayer(activation_fn, activation_prime), "Activation Layer")

    net.add(LinearLayer(int(hidden_length / 2), outputs), name="Output")
    net.add(ActivationLayer(sigmoid, sigmoid_prime), "Final Activation Layer")

    return net
