from layers import *
from functions import *

def build_neural_net(features, outputs, hidden_length, activation_fn, activation_prime):
    net = NeuralNet(mse, mse_prime)


    net.add(LinearLayer(features, hidden_length), name = "Input")
    net.add(ActivationLayer(activation_fn, activation_prime), "Activation Layer 1")

    net.add(LinearLayer(hidden_length, hidden_length), name = "Hidden Layer 1")
    net.add(ActivationLayer(activation_fn, activation_prime), "Activation Layer 2")

    net.add(LinearLayer(hidden_length,2 * hidden_length), name = "Hidden Layer 2")
    net.add(ActivationLayer(activation_fn, activation_prime), "Activation Layer 3")

    net.add(LinearLayer(2 * hidden_length, 4 * hidden_length), name = "Hidden Layer 3")
    net.add(ActivationLayer(activation_fn, activation_prime), "Activation Layer 4")

    net.add(LinearLayer(4 * hidden_length, 2 * hidden_length), name = "Hidden Layer 4")
    net.add(ActivationLayer(activation_fn, activation_prime), "Activation Layer 5")

    net.add(LinearLayer(2 * hidden_length, hidden_length), name = "Hidden Layer 5")
    net.add(ActivationLayer(activation_fn, activation_prime), "Activation Layer 6")

    net.add(LinearLayer(hidden_length, int(hidden_length / 2)), name = "Hidden Layer 6")
    net.add(ActivationLayer(activation_fn, activation_prime), "Activation Layer 7")
    
    net.add(LinearLayer(int(hidden_length / 2), int(hidden_length / 2)), name = "Hidden Layer 7")
    net.add(ActivationLayer(activation_fn, activation_prime), "Activation Layer 8")

    net.add(LinearLayer(int(hidden_length / 2), int(hidden_length / 2)), name = "Hidden Layer 8")
    net.add(ActivationLayer(activation_fn, activation_prime), "Activation Layer 9")

    net.add(LinearLayer(int(hidden_length / 2), outputs), name="Hidden Layer 9")
    net.add(ActivationLayer(sigmoid, sigmoid_prime), "Output")

    return net

def build_siren_net(features, outputs, hidden_length):
    net = NeuralNet(binary_cross_entropy, binary_cross_entropy_prime)
    net.add(SinusoidalLayer(features, hidden_length, is_first=True), name="SIREN Layer 1")

    net.add(SinusoidalLayer(hidden_length, hidden_length), name="SIREN Layer 2")
    net.add(SinusoidalLayer(hidden_length, 2 * hidden_length), name="SIREN Layer 3")
    net.add(SinusoidalLayer(2 * hidden_length, hidden_length), name="SIREN Layer 4")
    net.add(SinusoidalLayer(hidden_length, int(hidden_length / 2)), name="SIREN Layer 5")
    net.add(SinusoidalLayer(int(hidden_length / 2), int(hidden_length / 2)), name="SIREN Layer 6")
    net.add(SinusoidalLayer(int(hidden_length / 2), int(hidden_length / 2)), name="SIREN Layer 7")

    net.add(LinearLayer(int(hidden_length / 2), outputs), name="Output Linear")
    net.add(ActivationLayer(sigmoid, sigmoid_prime), "Final Activation Layer")

    return net

