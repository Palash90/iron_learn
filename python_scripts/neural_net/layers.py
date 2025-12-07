import cupy as np;
import time;
import math;
import numpy

class LinearLayer:
    def __init__(self, inputSize, outputSize):
        self.weights = np.random.randn(inputSize, outputSize, dtype=np.float32)
        self.biases = np.random.randn(1, outputSize, dtype=np.float32) 
        
    def forward(self, input):
        self.input = input
        self.output = self.input @ self.weights + self.biases 
        return self.output
    
    def backward(self, error, lr):
        input_error = error @ self.weights.T 
        weights_error = self.input.T @ error 
        
        biases_error = np.sum(error, axis=0, keepdims=True)
        
        self.weights -= lr * weights_error
        self.biases -= lr * biases_error 
        return input_error
    
class ActivationLayer:
    def __init__(self, activation, fPrime):
        self.activation = activation
        self.fPrime = fPrime
    
    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output
    
    def backward(self, error, lr):
        return self.fPrime(self.input) * error

class NeuralNet:
    def __init__(self, loss, loss_prime):
        self.layers = []
        self.loss = loss
        self.loss_prime = loss_prime
        self.layer_info = []

    def add(self, layer, name):
        self.layers.append(layer)
        self.layer_info.append({'layer': layer, 'name': name, 'type': layer.__class__.__name__})

    def predict(self, input_data):
        output = input_data 
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self, x_train, y_train, epochs, epoch_offset, learning_rate, hook):
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)

        start_time = time.time()

        epoch_error = []

        lr_min = 1e-6 
        lr_max = learning_rate
        

        for i in range(epochs):
            hook(self, i)

            decay_factor = 0.5 * (1 + math.cos(math.pi * i / (epochs+epoch_offset)))
            current_lr = lr_min + (lr_max - lr_min) * decay_factor    
            
            output = x_train
            for layer in self.layers:
                output = layer.forward(output)

            err = self.loss(y_train, output)

            error = self.loss_prime(y_train, output)
            for layer in reversed(self.layers):
                error = layer.backward(error, current_lr)
            
            epoch_error.append([i, err])

            if i == 0 or (i + 1) % 1000 == 0:
                np.cuda.runtime.deviceSynchronize() 
        
        np.cuda.runtime.deviceSynchronize()
        end_time = time.time()
        self.save_weights('final_model_weights.npz')
        print(f"\nTraining completed in {end_time - start_time:.4f} seconds.")

    def save_weights(self, filepath):
        state_dict = {}
        for i, info in enumerate(self.layer_info):
            if isinstance(info['layer'], LinearLayer):
                layer = info['layer']
                state_dict[f'W{i}_{info["name"]}'] = layer.weights.get() 
                state_dict[f'B{i}_{info["name"]}'] = layer.biases.get()
        
        numpy.savez_compressed(filepath, **state_dict)
        print(f"✅ Weights and biases saved to {filepath}")

    
    def load_weights(self, filepath):
        """Loads weights and biases into the Linear Layers."""
        try:
            loaded_data = numpy.load(filepath, allow_pickle=True)
            linear_layer_index = 0
            
            for i, info in enumerate(self.layer_info):
                if isinstance(info['layer'], LinearLayer):
                    layer = info['layer']
                    w_key = f'W{i}_{info["name"]}'
                    b_key = f'B{i}_{info["name"]}'
                    
                    if w_key in loaded_data and b_key in loaded_data:
                        loaded_weights = np.asarray(loaded_data[w_key])
                        loaded_biases = np.asarray(loaded_data[b_key])
                        
                        if layer.weights.shape == loaded_weights.shape and layer.biases.shape == loaded_biases.shape:
                             layer.weights = loaded_weights
                             layer.biases = loaded_biases
                             print(f"Loaded weights for {info['name']} (Layer {i})")
                        else:
                             print(f"❌ Dimension mismatch for {info['name']} (Layer {i}). Skipping load.")
                             
                    else:
                        print(f"❌ Key missing for {info['name']} (Layer {i}). Skipping load.")
                        
                    linear_layer_index += 1
            
            print(f"✅ Weights and biases loaded from {filepath}")
            return True
            
        except FileNotFoundError:
            print(f"❌ Error: Weights file not found at {filepath}")
            return False
        except Exception as e:
            print(f"❌ An error occurred during loading: {e}")
            return False
  