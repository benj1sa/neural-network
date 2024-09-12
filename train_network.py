import numpy as np

def trainNetwork(n, output_nodes, epochs):
    # Load the MNIST training data file and read data into a list
    training_data_file = open("mnist_dataset/mnist_train.csv", "r")
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # Training the Neural Network

    for e in range(epochs):
        for record in training_data_list:
            # Split the record of long text string of comma separated values into individual values
            # Additionally, convert types from char -> int
            all_values = [int(char) for char in record.split(",")]

            # Prepare the input by converting the color values from 0 - 255 to 0.01 - 1.0
            inputs = (np.asarray(all_values[1:]) / 255.0 * 0.99) + 0.01

            # set target values
            # all_values[0] is the target number for this record
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99

            n.train(inputs, targets)

            pass

    pass