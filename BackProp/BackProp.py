from random import uniform
import math

class NeuralNetwork():
    def __init__(self):
        self.layers = []
        self.neurons = [[]]
        self.biases = [[]]
        self.weights = [[[]]]
        self.activations = []

        self.fitness = 0

        self.learningRate = 0.01
        self.cost = 0

    def NetworkInit(self, layers, layerActivations):
        self.layers = layers
        self.activations = list(range(len(self.layers) - 1))

        for i in range(len(self.layers) - 1):
            action = layerActivations[i]
            if action == "sigmoid":
                self.activations[i] = 0
            elif action == "tanh":
                self.activations[i] = 1
            elif action == "relu":
                self.activations[i] = 2
            elif action == "leakyrelu":
                self.activations[i] = 3
            elif action == "stepped":
                self.activations[i] = 4
            else:
                self.activations[i] = 2

        self.InitNeurons()
        self.InitBiases()
        self.InitWeights()

    def InitNeurons(self):
        self.neurons.clear()
        for i in range(len(self.layers)):
            self.neurons.append(list(range(self.layers[i])))

    def InitBiases(self):
        self.biases.clear()
        for i in range(1, len(self.layers)):
            bias = []
            for j in range(self.layers[i]):
                bias.append(uniform(-0.5, 0.5))
            self.biases.append(bias)

    def InitWeights(self):
        self.weights.clear()
        for i in range(1, len(self.layers)):
            layerWeightsList = [[]]
            layerWeightsList.clear()
            neuronsInPreviousLayer = self.layers[i - 1]
            for j in range(self.layers[i]):
                neuronWeights = []
                for k in range(neuronsInPreviousLayer):
                    neuronWeights.append(uniform(-0.5, 0.5))
                layerWeightsList.append(neuronWeights)
            self.weights.append(layerWeightsList)

    def Feed(self, inputs):
        for i in range(len(inputs)):
            self.neurons[0][i] = inputs[i]
        for i in range(1, len(self.layers)):
            layer = i - 1
            for j in range(self.layers[i]):
                value = 0
                for k in range(self.layers[i - 1]):
                    value += self.weights[i -
                                          1][j][k] * self.neurons[i - 1][k]
                self.neurons[i][j] = self.activate(
                    value + self.biases[i - 1][j], layer)
        return self.neurons[len(self.layers) - 1]

    def activate(self, value, layer):
        if self.activations[layer] == 0:
            return self.sigmoid(value)
        elif self.activations[layer] == 1:
            return self.tanh(value)
        elif self.activations[layer] == 2:
            return self.relu(value)
        elif self.activations[layer] == 3:
            return self.leakyrelu(value)
        elif self.activations[layer] == 4:
            return self.stepped(value)
        else:
            return self.relu(value)

    def activateDer(self, value, layer):
        if self.activations[layer] == 0:
            return self.sigmoidDer(value)
        if self.activations[layer] == 1:
            return self.tanhDer(value)
        if self.activations[layer] == 2:
            return self.reluDer(value)
        if self.activations[layer] == 3:
            return self.leakyreluDer(value)
        if self.activations[layer] == 4:
            return self.steppedDer(value)
        else:
            return self.reluDer(value)

    def sigmoid(self, x):
        k = math.exp(x)
        return k / (1 + k)

    def tanh(self, x):
        return math.tanh(x)

    def relu(self, x):
        if 0 >= x:
            return 0
        else:
            return x

    def leakyrelu(self, x):
        if 0 >= x:
            return 0.01 * x
        else:
            return x

    def stepped(self, x):
        if x < 0:
            return 0
        elif x > 0:
            return 1
        else:
            return 0.5

    def sigmoidDer(self, x):
        return x * (1 - x)

    def tanhDer(self, x):
        return 1 - (x * x)

    def reluDer(self, x):
        if 0 >= x:
            return 0
        else:
            return 1

    def leakyreluDer(self, x):
        if 0 >= x:
            return 0.01
        else:
            return 1

    def steppedDer(self, x):
        if x < 0:
            return 0
        elif x > 0:
            return 1
        else:
            return 0.5

    def BackPropagate(self, output, expected):
        self.cost = 0
        for i in range(len(output) - 1):
            self.cost += math.pow(output[i] - expected[i], 2)
        self.cost /= 2

        gamma = [[]]
        gamma.clear()

        for i in range(len(self.layers)):
            gamma.append(list(range(self.layers[i])))

        layer = len(self.layers) - 2
        for i in range(len(output)):
            gamma[len(self.layers) - 1][i] = (output[i] - expected[i]
                                         ) * self.activateDer(output[i], layer)
        for i in range(self.layers[len(self.layers) - 1]):
            self.biases[len(
                self.layers) - 2][i] -= gamma[len(self.layers) - 1][i] * self.learningRate
            for j in range(self.layers[len(self.layers) - 2]):
                self.weights[len(self.layers) - 2][i][j] -= gamma[len(self.layers) - 1][i] * \
                    self.neurons[len(self.layers) - 2][j] * self.learningRate

        for i in range(len(self.layers) - 2, i > 0, -1):
            layer = i - 1
            for j in range(self.layers[i]):
                gamma[i][j] = 0
                for k in range(len(gamma[i + 1])):
                    gamma[i][j] += gamma[i + 1][k] * self.weights[i][k][j]
                gamma[i][j] *= self.activateDer(self.neurons[i][j], layer)
            for j in range(self.layers[i]):
                self.biases[i-1][j] -= gamma[i][j] * self.learningRate
                for k in range(self.layers[i-1]):
                    self.weights[i - 1][j][k] -= gamma[i][j] * \
                        self.neurons[i - 1][k] * self.learningRate

    def Mutate(self, high, val):
        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                if uniform(0, high) <= 2:
                    self.biases[i][j] += uniform(-val, val)
                else:
                    self.biases[i][j] = self.biases[i][j]

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    if uniform(0, high) <= 2:
                        self.weights[i][j][k] += uniform(-val, val)
                    else:
                        self.weights[i][j][k] = self.weights[i][j][k]

    def copy(self, nn):
        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                nn.biases[i][j] = self.biases[i][j]
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    nn.weights[i][j][k] = self.weights[i][j][k]
        return nn

    def Load(self, path):
        with open(path, "r") as file:
            ListLines = file.split("\n")
            print(ListLines)
            index = 1

            if len(ListLines) > 0:
                for i in range(len(self.biases)):
                    for j in range(len(self.biases[i])):
                        self.biases[i][j] = float(ListLines[index])
                        index += 1

                for i in range(len(self.weights)):
                    for j in range(len(self.weights[i])):
                        for k in range(len(self.weights[i][j])):
                            self.weights[i][j][k] = float(ListLines[index])
                            index += 1

    def Save(self, path):
        with open(path, "w") as file:
            for i in range(len(self.biases)):
                for j in range(len(self.biases[i])):
                    file.write(self.biases[i][j])

            for i in range(len(self.weights)):
                for j in range(len(self.weights[i])):
                    for k in range(len(self.weights[i][j])):
                        file.write(self.weights[i][j][k])