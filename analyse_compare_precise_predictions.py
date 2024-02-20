import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.models import load_model, Model

data = pd.read_csv("person_2nd/1602.csv")
data = np.array([data])
inputs = data[:, :, 1:][:, :, :-1].astype(np.float32)
true_outputs = data[:, :, -1:].astype(np.float32)

model = load_model("Momen_person_champion.keras")

# Assuming you have your inputs and true outputs in variables `inputs` and `true_outputs`
# And your model in variable `model`

predicted_outputs = model.predict(inputs)

print(predicted_outputs)

# Initialize an empty list to store inputs where predicted output matches true output
matching_inputs = []

for i in range(len(predicted_outputs[0])):
    # If the predicted output matches the true output
    if np.allclose(predicted_outputs[0][i][0], true_outputs[0][i][0], atol=0.1):
        # Store the corresponding input
        matching_inputs.append(inputs[0][i])

# Now, `matching_inputs` contains the inputs where the predicted output matches the true output
print(matching_inputs)
print(len(matching_inputs))
print(len(inputs[0]))


# Convert matching_inputs to a numpy array for easier manipulation
matching_inputs_np = np.array(matching_inputs)

# For each dimension in the input data
for i in range(matching_inputs_np.shape[1]):
    # Create a new figure
    plt.figure()

    # Plot a histogram of the data in the current dimension
    plt.hist(matching_inputs_np[:, i], bins=20)

    # Set the title of the plot
    plt.title(f'Input Dimension {i + 1}')

    # Show the plot
    # plt.show()
    plt.savefig(f"precise_prediction_chart/matching Input Dimension {i + 1}.png")

    # close the plot
    plt.close()


# Initialize an empty list to store inputs where predicted output does not match true output
non_matching_inputs = []

for i in range(len(predicted_outputs[0])):
    # If the predicted output does not match the true output
    if not np.allclose(predicted_outputs[0][i][0], true_outputs[0][i][0], atol=0.1):
        # Store the corresponding input
        non_matching_inputs.append(inputs[0][i])

# Now, `non_matching_inputs` contains the inputs where the predicted output does not match the true output

# Convert non_matching_inputs to a numpy array for easier manipulation
non_matching_inputs_np = np.array(non_matching_inputs)

# For each dimension in the input data
for i in range(non_matching_inputs_np.shape[1]):
    # Create a new figure
    plt.figure()

    # Plot a histogram of the data in the current dimension
    plt.hist(non_matching_inputs_np[:, i], bins=20)

    # Set the title of the plot
    plt.title(f'Input Dimension {i+1}')

    # Show the plot
    # plt.show()
    plt.savefig(f"precise_prediction_chart/non_matching Input Dimension {i + 1}.png")

    # close the plot
    plt.close()
