import math


class Layer:
    def __init__(self, input_size, output_size, activation_function='None', lr=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.weight = [[0.01 for i in range(output_size)] for j in range(input_size)]
        self.bias = [0.01 for i in range(output_size)]
        self.output = [0.01 for i in range(output_size)]
        self.learning_rate = lr

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-1 * x))

    def forward(self, input_data):
        """
            forward network activation

            Parameters
            ----------
            input_data: list
                a train data

            Returns
            -------
            output: list
                two dimension vector representing the belief of data being in class a and b
        """
        tmp_output = []
        for i in range(self.output_size):
            s = 0
            for j in range(self.input_size):
                s += input_data[j] * self.weight[j][i]
            s += self.bias[i]
            if self.activation_function == 'sigmoid':
                tmp_output.append(self.sigmoid(s))
            else:
                tmp_output.append(s)
            self.output = tmp_output
        return self.output

    def backward(self, err_list):
        """
            backward error propagation

            Parameters
            ----------
            err_list: list
                the error past from the next layer

            Returns
            -------
            err_list_next: list
                the error of the neural cells in this layer
        """
        err_list_next = [0 for i in range(self.input_size)]

        if len(err_list) == 2:
            for i in range(self.output_size):
                err = self.output[i] * (1 - self.output[i]) * err_list[i]
                for j in range(self.input_size):
                    err_list_next[j] += self.weight[j][i] * err
                    self.weight[j][i] += self.learning_rate * err * self.output[i]
                self.bias[i] += self.learning_rate * err
        else:
            for i in range(self.output_size):
                err = self.output[i] * (1 - self.output[i]) * err_list[i]
                for j in range(self.input_size):
                    err_list_next[j] += self.weight[j][i] * err
                    self.weight[j][i] += self.learning_rate * err * self.output[i]
                self.bias[i] += self.learning_rate * err
        return err_list_next