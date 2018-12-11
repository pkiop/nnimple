import numpy as np

class DeepNeuralNetwork:
    def __init__(self, input_layers, hidden_layers, output_layers, test_data):
        self.inputs = input_layers
        self.hiddens = hidden_layers
        self.outputs = output_layers
        self.test_data = test_data
        self.w_ih = np.random.randn(self.inputs, self.hiddens) / np.sqrt(self.inputs/2)
        self.w_ho = np.random.randn(self.hiddens, self.outputs) / np.sqrt(self.hiddens/2)
        self.data = None


    def predict(self, x):
        data = self.normalize(np.asfarray(x.split(',')))

        data = data[1:]

        layer_1 = self.sigmoid(np.dot(data, self.w_ih))
        output = self.sigmoid(np.dot(layer_1, self.w_ho))
        return output

    def train(self, training_data, lr=0.1, epoch=1):
        for ech in range(0, epoch):
            for i, x in enumerate(training_data):
                target = np.array(np.zeros(self.outputs) + 0.01, ndmin=2)
                target[0][int(x[0])] = 0.99
                x = self.normalize(np.asfarray(x.split(",")))

                l1 = self.sigmoid(np.dot(x[1:], self.w_ih))
                l2 = self.sigmoid(np.dot(l1, self.w_ho))

                l2_e = (target - l2) * (l2 * (1 - l2))
                l1_e = l2_e.dot(self.w_ho.T) * (l1 * (1 - l1))

                self.w_ho = self.w_ho + lr * l2_e.T.dot(np.array(l1, ndmin=2)).T
                self.w_ih = self.w_ih + lr * l1_e.T.dot(np.array(x[1:], ndmin=2)).T

                if i % 20000000000 == 0 :
                    self.print_accuracy()

    def print_accuracy(self):
        matched = 0

        for x in self.test_data:
            label = int(x[0])
            predicted = np.argmax(self.predict(x))
            if label == predicted :
                matched = matched + 1
        print('ACC : {0}%'.format(100*matched/len(self.test_data)))

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def normalize(self, x):
        return (x / 255.0) * 0.99 + 0.01


data_file = open("data/image.csv", "r")
print(data_file)
training_data = data_file.readlines()
data_file.close()

test_data_file = open("data/image.csv", "r")
test_data = test_data_file.readlines()
test_data_file.close()
print(type(test_data))

network = DeepNeuralNetwork(input_layers=256, hidden_layers=100, output_layers=10, test_data=test_data)
network.train(training_data, lr=0.01, epoch=100)
