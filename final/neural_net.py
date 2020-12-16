from nn_layers import *


def mse(output, real):
    s = 0
    for i in range(len(output)):
        s += math.pow((output[i] - real[i]), 2)
    s /= len(output)
    return s


def nn(train_a, train_b, test, iter_num):
    """
        the neural network

        Parameters
        ----------
        train_a: list
            all the train data of label a
        train_b: list
            all the train data of label b
        test: list
            all the test data

        Returns
        -------
        label: list
            the label of the test data
    """
    train_a_flatten = [[val for sublist in train_a[i] for val in sublist] for i in range(len(train_a))]
    train_b_flatten = [[val for sublist in train_b[i] for val in sublist] for i in range(len(train_b))]
    test_flatten = [[val for sublist in test[i] for val in sublist] for i in range(len(test))]

    layer1 = Layer(25, 10, 'sigmoid')
    layer2 = Layer(10, 5, 'sigmoid')
    layer3 = Layer(5, 2, 'sigmoid')

    err_list = [0, 0]
    for iter_cnt in range(iter_num):
        for i in range(len(train_a_flatten)):
            output1 = layer1.forward(train_a_flatten[i])
            output2 = layer2.forward(output1)
            output3 = layer3.forward(output2)
            err_list[0] = 1 - output3[0]
            err_list[1] = 0 - output3[1]
            err_list3 = layer3.backward(err_list)
            err_list2 = layer2.backward(err_list3)
            layer1.backward(err_list2)

            output1 = layer1.forward(train_b_flatten[i])
            output2 = layer2.forward(output1)
            output3 = layer3.forward(output2)
            err_list[0] = 0 - output3[0]
            err_list[1] = 1 - output3[1]
            err_list3 = layer3.backward(err_list)
            err_list2 = layer2.backward(err_list3)
            layer1.backward(err_list2)

        if iter_cnt % 500 == 0:
            print("iter: ", iter_cnt)
    label = []
    for i in range(len(test_flatten)):
        output1 = layer1.forward(test_flatten[i])
        output2 = layer2.forward(output1)
        output3 = layer3.forward(output2)

        if output3[0] > output3[1]:
            label.append('A')
        else:
            label.append('B')
    return label