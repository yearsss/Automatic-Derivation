import random
import math

class Neuron:
    def __init__(self, weight, bia):
        self.bia = bia
        self.weight = weight
        self.net_output = 0
        self.error = 0
        self.output = 0

class NeuronLayer:
    
    def __init__(self, num_neurons, weights, bias):
        """
            bias: layer bias array
            weights: 2d-array
        """
        self.neurons = []
       
        for i in range(num_neurons):
            if weights and bias:
                self.neurons.append(Neuron(weights[i], bias[i]))
            else:
                self.neurons.append(Neuron(None, None))

            
    def forward(self, input_layer):    
        for i in range(len(self.neurons)):
            self.neurons[i].net_output = 0.
            for j in range(len(input_layer.neurons)):
                self.neurons[i].net_output += \
                    input_layer.neurons[j].output * self.neurons[i].weight[j]
               
            self.neurons[i].output = 1 / (1 + math.exp(-self.neurons[i].net_output + self.neurons[i].bia))
        
    def backward(self, output_layer):
        for i in range(len(self.neurons)):
            self.neurons[i].error = 0.
            for j in range(len(output_layer.neurons)):
                self.neurons[i].error += \
                    output_layer.neurons[j].error * output_layer.neurons[j].weight[i] \
                    * self.neurons[i].output * (1 - self.neurons[i].output)
                
            
    

class NeuralNetwork:
    LEARNING_RATE = 0.5
    def __init__(self, num_inputs, num_outputs, num_hidden, 
                 hidden_layer_weights = None, hidden_layer_bias = None,
                 output_layer_weights = None, output_layer_bias = None):
        
        self.input_layer = NeuronLayer(num_inputs, None, None)
        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_weights, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_weights, output_layer_bias)
        
    def inspect(self):
        print 'hidden_layer weight: {}'.format([neuron.weight for neuron in self.hidden_layer.neurons])
        print 'output_layer weight: {}'.format([neuron.weight for neuron in self.output_layer.neurons])
        
        print 'hidden_layer output: {}'.format([neuron.output for neuron in self.hidden_layer.neurons])
        print 'output_layer output: {}'.format([neuron.output for neuron in self.output_layer.neurons])
        
        print 'hidden_layer error: {}'.format([neuron.error for neuron in self.hidden_layer.neurons])
        print 'output_layer error: {}'.format([neuron.error for neuron in self.output_layer.neurons])
    def forward(self):
        self.hidden_layer.forward(self.input_layer)
        self.output_layer.forward(self.hidden_layer)
        
    def backward(self):
        self.hidden_layer.backward(self.output_layer)
        self.input_layer.backward(self.hidden_layer)

    def train(self, training_inputs, training_outputs):
        
        # input 
        for i in range(len(self.input_layer.neurons)):
            self.input_layer.neurons[i].output = training_inputs[i]
        self.forward()
        
        total_error = 0
        size = len(self.output_layer.neurons)
        for i in range(size):
            total_error += ((training_outputs[i] - self.output_layer.neurons[i].output) ** 2) / (size * 2)
                
        print "total_error: {}".format(total_error)
        # 1. Output neuron deltas
        for i in range(size):
            # ∂E/∂z 
            self.output_layer.neurons[i].error = (self.output_layer.neurons[i].output - training_outputs[i]) \
                                            * self.output_layer.neurons[i].output \
                                            * (1 - self.output_layer.neurons[i].output) 
        self.backward()
        
        # update output layer param
        for i in range(len(self.output_layer.neurons)):
            for j in range(len(self.hidden_layer.neurons)):         
                self.output_layer.neurons[i].weight[j] -= \
                        self.LEARNING_RATE * self.output_layer.neurons[i].error * self.hidden_layer.neurons[j].output
                
                    
        # update hidden layer param
        for i in range(len(self.hidden_layer.neurons)):
            for j in range(len(self.input_layer.neurons)):
                self.hidden_layer.neurons[i].weight[j] -= \
                        self.LEARNING_RATE * self.hidden_layer.neurons[i].error * self.input_layer.neurons[j].output
                

nn = NeuralNetwork(2, 2, 2, hidden_layer_weights=[[0.2, 0.2], [0.2, 0.3]], hidden_layer_bias=[.0, .0],
                  output_layer_weights=[[0.4, 0.5], [0.5, 0.5]], output_layer_bias=[.0 ,.0])

for i in range(100):
    # nn.inspect()
    nn.train([0.5, 0.1], [0.1, 0.9])
    # 
