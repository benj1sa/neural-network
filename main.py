from neural_network import neuralNetwork
from train_network import trainNetwork
import numpy as np
from PIL import Image

img_url = "number-images/messy/messy4.png"
img = Image.open(img_url).convert("L")
img_array = np.array(img)
img_data = 255.0 - img_array.reshape(784)
img_data = (img_data / 255.0 * 0.99) + 0.01

# Initialize the neural network
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Train the network with specified output_nodes and 3 epochs
trainNetwork(n, output_nodes, 3)
response = np.argmax(n.query(img_data))
print("network's answer: ", response)