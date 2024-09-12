from neural_network import neuralNetwork
from train_network import trainNetwork
import numpy as np

# Initialize the neural network
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.3
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Train the network
trainNetwork(n, output_nodes, 3)

# Test the neural network
test_data_file = open("mnist_dataset/mnist_test.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

# Initialize a scorecard for how well the network performs
scorecard = []
number_correct = 0

for record in test_data_list:
    # Split the record of long text string of comma separated values into individual values
    # Additionally, convert types from char -> int
    all_values = [int(char) for char in record.split(",")]

    # the correct answer is the first value
    correct_label = int(all_values[0])
    print(correct_label, "correct label")

    # Prepare the input by converting the color values from 0 - 255 to 0.01 - 1.0
    inputs = (np.asarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    print(label, "network's answer")

    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
        number_correct += 1
    else:
        # network's answer didn't match correct answer, add 0 to scorecard
        scorecard.append(0)

    pass

# Print the percentage correct 
print("Accuracy: ",100 * (number_correct / len(scorecard)), "%")