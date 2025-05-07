#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import autokeras as ak
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score,  roc_curve
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

#!/usr/bin/env python
# coding: utf-8



print(tf.__version__)




get_ipython().system('pip install autokeras')








data=pd.read_csv("benchmark_dataset.csv")
print(data.head(2))




# from sklearn.preprocessing import LabelBinarizer

# Function to one-hot encode sgRNA sequences
def one_hot_encode_sequences(sequences):
    # Initialize the LabelBinarizer to encode the four nucleotides
    lb = LabelBinarizer()
    lb.fit(list('ACGT'))

    # Apply one-hot encoding to each sequence
    encoded_sequences = []
    for seq in sequences:
        encoded_seq = lb.transform(list(seq))  # Transform each nucleotide in the sequence
        encoded_sequences.append(encoded_seq.flatten())  # Flatten to make it a single array per sequence

    return encoded_sequences

# Apply one-hot encoding to the sgRNA sequences in the dataset
encoded_data = one_hot_encode_sequences(data['sgRNA'])

# Convert the list to a DataFrame for easier handling
encoded_df = pd.DataFrame(encoded_data)
print(encoded_df.head(2))
encoded_df['Efficacy'] = data['Efficacy']  # Add the efficacy labels back to the DataFrame
encoded_df.head(3)





print(encoded_df.shape)


# #Convert DataFrame to Numpy Array



X = encoded_df.drop('Efficacy', axis=1).to_numpy()
y = encoded_df['Efficacy'].to_numpy()




# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)  # 70% training, 30% testing




print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)




timesteps = 10
features = 9  # 92 features divided by 10 timesteps

# Truncate the last few features to make it divisible by timesteps
X_train = X_train[:, :timesteps * features]
X_test = X_test[:, :timesteps * features]

# Reshape
X_train = X_train.reshape((X_train.shape[0], timesteps, features))
X_test=X_test.reshape((X_test.shape[0], timesteps, features))


# # Variant 1: Standard RNN with Dense Block




# Step 1: Define the input node
input_node = ak.Input()  # Generic input for various data types

# Step 2: Add an RNN layer
output_node = ak.RNNBlock()(input_node)  # AutoKeras selects the best RNN (LSTM/GRU)

# Step 3: Add a Dense block
output_node = ak.DenseBlock()(output_node)  # Fully connected layers

# Step 4: Define the output layer for classification
output_node = ak.ClassificationHead()(output_node)

# Step 5: Define the AutoKeras model
model = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=20, overwrite=True)

# Step 6: Fit the model
model.fit(X_train, y_train, epochs=10)





# Set up parameters
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []
auc_scores = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Define the AutoKeras model
    input_node = ak.Input()
    output_node = ak.RNNBlock()(input_node)  # First RNN layer
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.ClassificationHead()(output_node)
    model = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=20, overwrite=True)
    
    # Fit the model
    model.fit(X_train_fold, y_train_fold, epochs=10, verbose=0)
    
    # Evaluate the model
    y_val_pred = model.predict(X_val_fold)
    y_val_prob = model.predict(X_val_fold, batch_size=32).flatten()
    
    # Calculate metrics
    val_accuracy = accuracy_score(y_val_fold, y_val_pred)
    auc_roc = roc_auc_score(y_val_fold, y_val_prob)
    
    accuracy_scores.append(val_accuracy)
    auc_scores.append(auc_roc)

# Calculate mean and standard deviation of accuracy and AUC-ROC
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)

print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")
print(f"Mean AUC-ROC: {mean_auc:.4f}")
print(f"Standard Deviation of AUC-ROC: {std_auc:.4f}")


# # Variant 2: Stacked RNN Layers




# Step 1: Define the input node
input_node = ak.Input()

# Step 2: Add the first RNN layer (set return_sequences=True to pass sequences to the next RNN layer)
rnn_output = ak.RNNBlock(return_sequences=True)(input_node)  # First RNN layer

# Step 3: Add a second RNN layer (can now process the sequence output from the previous layer)
rnn_output = ak.RNNBlock()(rnn_output)  # Second RNN layer (final RNN block)

# Step 4: Add a Dense block
output_node = ak.DenseBlock()(rnn_output)

# Step 5: Define the output layer for classification
output_node = ak.ClassificationHead()(output_node)

# Step 6: Define the AutoKeras model
model = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=20, overwrite=True)

# Step 7: Fit the model
model.fit(X_train, y_train, epochs=10)





# Set up parameters
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []
auc_scores = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Define the AutoKeras model
    input_node = ak.Input()
    rnn_output = ak.RNNBlock(return_sequences=True)(input_node)  # First RNN layer
    rnn_output = ak.RNNBlock()(rnn_output)  # Second RNN layer (final RNN block)
    output_node = ak.DenseBlock()(rnn_output)
    output_node = ak.ClassificationHead()(output_node)
    model = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=20, overwrite=True)
    
    # Fit the model
    model.fit(X_train_fold, y_train_fold, epochs=10, verbose=0)
    
    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    accuracy_scores.append(val_accuracy)
    
    # Predict probabilities
    y_val_prob = model.predict(X_val_fold)
    
    # Check if the prediction is for binary classification
    if y_val_prob.shape[1] == 2:
        # Ensure y_val_prob is the probability of the positive class
        y_val_prob = y_val_prob[:, 1]
    else:
        y_val_prob = y_val_prob[:, 0]  # In case the model only outputs one probability
    
    # Calculate AUC-ROC
    auc_roc = roc_auc_score(y_val_fold, y_val_prob)
    auc_scores.append(auc_roc)

# Calculate mean and standard deviation of accuracy and AUC-ROC
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)

print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")
print(f"Mean AUC-ROC: {mean_auc:.4f}")
print(f"Standard Deviation of AUC-ROC: {std_auc:.4f}")







# # Variant 3: RNN with Dropout



# import tensorflow as tf
# from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Input
# from tensorflow.keras.models import Model

# # Step 1: Define the input node
# input_node = Input(shape=(timesteps, features))  # specify the shape

# # Step 2: Add a bidirectional RNN (Bidirectional LSTM)
# rnn_output = Bidirectional(LSTM(128))(input_node)  # LSTM with 128 units wrapped in Bidirectional

# # Step 3: Add Dropout layer
# dropout_output = Dropout(0.5)(rnn_output)

# # Step 4: Add a Dense layer for classification
# dense_output = Dense(64, activation='relu')(dropout_output)
# output_node = Dense(num_classes, activation='softmax')(dense_output)

# # Step 5: Define and compile the model
# model = Model(inputs=input_node, outputs=output_node)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Step 6: Train the model
# model.fit(X_train, y_train, epochs=10)

# # This avoids mixing libraries and ensures compatibility


# # Cross-validation in RNN with dropout




# Set up parameters
timesteps = 10
features = 9
num_classes = 2  # Update with the actual number of classes

# Reshape your data to fit your model's input
X_train = X_train[:, :timesteps * features].reshape((X_train.shape[0], timesteps, features))
X_test = X_test[:, :timesteps * features].reshape((X_test.shape[0], timesteps, features))

# Prepare for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Define the model
    input_node = Input(shape=(timesteps, features))
    rnn_output = Bidirectional(LSTM(128))(input_node)
    dropout_output = Dropout(0.5)(rnn_output)
    dense_output = Dense(64, activation='relu')(dropout_output)
    output_node = Dense(num_classes, activation='softmax')(dense_output)
    
    model = Model(inputs=input_node, outputs=output_node)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train_fold, y_train_fold, epochs=10, verbose=0)
    
    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    accuracy_scores.append(val_accuracy)

# Calculate mean and standard deviation of accuracy
mean_accuracy = np.mean(accuracy_scores)
std_deviation = np.std(accuracy_scores)

print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation: {std_deviation:.4f}")




# Evaluate the model
accuracy_scores = []
auc_scores = [] 
val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
accuracy_scores.append(val_accuracy)
  
  # Predict probabilities
y_val_prob = model.predict(X_val_fold)[:, 1]
  
  # Calculate AUC-ROC
auc_roc = roc_auc_score(y_val_fold, y_val_prob)
auc_scores.append(auc_roc)

# Calculate mean and standard deviation of accuracy and AUC-ROC
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)

print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")
print(f"Mean AUC-ROC: {mean_auc:.4f}")
print(f"Standard Deviation of AUC-ROC: {std_auc:.4f}")




# Retrieve the best model manually
best_model = model.tuner.get_best_model()

# Save the best model
best_model.save("NAS_RNN_dropout.h5")


# # Variant 4: RNN with Different Recurrent Units (GRU instead of LSTM)




# Set up parameters
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []
auc_scores = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Define the AutoKeras model with RNN and GRU
    input_node = ak.Input()
    rnn_output = ak.RNNBlock(layer_type="lstm")(input_node)  # Specify GRU
    dense_output = ak.DenseBlock()(rnn_output)
    output_node = ak.ClassificationHead()(dense_output)
    
    model = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=20, overwrite=True)
    
    # Fit the model
    model.fit(X_train_fold, y_train_fold, epochs=10, verbose=0)
    
    # Evaluate the model
    y_val_pred = model.predict(X_val_fold)
    y_val_prob = model.predict(X_val_fold, batch_size=32).flatten()
    
    # Calculate metrics
    val_accuracy = accuracy_score(y_val_fold, y_val_pred)
    auc_roc = roc_auc_score(y_val_fold, y_val_prob)
    
    accuracy_scores.append(val_accuracy)
    auc_scores.append(auc_roc)

# Calculate mean and standard deviation of accuracy and AUC-ROC
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)

print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")
print(f"Mean AUC-ROC: {mean_auc:.4f}")
print(f"Standard Deviation of AUC-ROC: {std_auc:.4f}")


# # Variant 5: Bidirectional RNN



# from tensorflow.keras.layers import Bidirectional, LSTM, Dense
# from tensorflow.keras import Input, Model
# import autokeras as ak

# # Number of classes
# num_classes = 2  # Replace with your actual number of classes

# # Step 1: Define the input node
# input_node = Input(shape=(timesteps, features))  # specify the shape

# # Step 2: Add a bidirectional RNN (Bidirectional LSTM)
# rnn_output = Bidirectional(LSTM(128))(input_node)  # LSTM with 128 units wrapped in Bidirectional

# # Step 3: Add a dense layer for classification
# dense_output = Dense(64, activation='relu')(rnn_output)
# output_node = Dense(num_classes, activation='softmax')(dense_output)

# # Step 4: Define and compile the tf.keras model
# model = Model(inputs=input_node, outputs=output_node)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Step 5: Train the model
# model.fit(X_train, y_train, epochs=10)

# # If you need to then use the trained model in an AutoKeras pipeline for further tuning,
# # you can load this model and provide it to AutoKeras for optimization.




# import tensorflow as tf
# from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input
# from tensorflow.keras.models import Model

# # Step 1: Define the input node (specify the input shape based on your data)
# input_node = Input(shape=(X_train.shape[1], X_train.shape[2]))

# # Step 2: Add a bidirectional LSTM layer
# rnn_output = Bidirectional(LSTM(128))(input_node)

# # Step 3: Add a Dense layer
# dense_output = Dense(64, activation='relu')(rnn_output)

# # Step 4: Define the output layer for classification (binary classification)
# output_node = Dense(1, activation='sigmoid')(dense_output)

# # Step 5: Create the Keras model
# model = Model(inputs=input_node, outputs=output_node)

# # Step 6: Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Step 7: Fit the model
# model.fit(X_train, y_train, epochs=10)




# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)




# from sklearn.metrics import roc_auc_score,  roc_curve

# # Get predicted probabilities for the positive class
# y_pred_prob = model.predict(X_test)

# # Check if y_test is binary (1-dimensional array)
# if len(y_test.shape) == 1:  
#     # y_test is already binary, no need to change
#     y_test_binary = y_test
# else:
#     # For multi-class (e.g., one-hot encoded), select the positive class column
#     y_test_binary = y_test[:, 1]

# # Calculate AUC-ROC
# auc_roc = roc_auc_score(y_test_binary, y_pred_prob)
# print("Test AUC-ROC:", auc_roc)

# # Optional: Plot ROC curve
# fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_prob)
# plt.figure()
# plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc_roc:.2f})')
# plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()
# # Save the ROC curve plot as an image file
# plt.savefig('roc_curve_variant3new .png')  # Save the plot
# plt.show()  # Display the plot





# Get predicted probabilities for the positive class (assuming binary classification)
y_pred_prob = model.predict(X_test)[:, 1]

# y_test should be binary (1-dimensional array)
if len(y_test.shape) == 1:  
    y_test_binary = y_test
else:
    y_test_binary = y_test[:, 1]

# Calculate AUC-ROC
auc_roc = roc_auc_score(y_test_binary, y_pred_prob)
print("Test AUC-ROC:", auc_roc)

# Optional: Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
# Save the ROC curve plot as an image file
plt.savefig('roc_curve_variant5.png')  # Save the plot
plt.show()  # Display the plot




# Export the best model after the search
best_model = model.export_model()

# Print the model summary to get information about the layers
best_model.summary()

# Get the optimizer used in the best model
optimizer = best_model.optimizer
print("Optimizer used:", optimizer)





# Step 1: Define the input node
input_node = ak.Input()  # Generic input for various data types

# Step 2: Add an RNN layer
output_node = ak.RNNBlock()(input_node)  # Adds an RNN layer (AutoKeras will choose LSTM/GRU)

# Step 3: Add a Dense block (optional)
output_node = ak.DenseBlock()(output_node)  # Adds fully connected layers

# Step 4: Define the output layer for classification
output_node = ak.ClassificationHead()(output_node)

# Step 5: Define the AutoKeras model
model = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=50, overwrite=True)  # Increased max_trials

# Step 6: Fit the model (replace 'x_train' and 'y_train' with actual data)
model.fit(X_train, y_train, epochs=10)

# Step 7: Export the best model from the AutoKeras search
try:
    best_model = model.export_model()
    # Step 8: Visualize the model architecture
    # (a) Print a detailed summary of the architecture
    best_model.summary()
    # (b) Plot the model architecture (save as an image)
    plot_model(best_model, to_file='rnn_model_architecture.png', show_shapes=True)
except IndexError:
    print("No valid model found. Please check the search process and data.")








# import numpy as np
# import matplotlib.pyplot as plt
# import autokeras as ak
# from tensorflow.keras.callbacks import EarlyStopping

# # Define a function to train and evaluate the model for different learning rates
# def evaluate_learning_rates(X_train, y_train, learning_rates):
#     results = []
    
#     for lr in learning_rates:
#         print(f"Training with learning rate: {lr:.2e}")
        
#         # Step 1: Define the AutoKeras model
#         input_node = ak.Input(shape=(timesteps, features))  # Replace with your actual shape
#         output_node = ak.RNNBlock()(input_node)
#         output_node = ak.DenseBlock()(output_node)
#         output_node = ak.ClassificationHead()(output_node)
        
#         model = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=10, overwrite=True)
        
#         # Step 2: Fit the model
#         early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
#         history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, callbacks=[early_stopping], verbose=1)
        
#         # Step 3: Store the validation loss
#         val_loss = history.history['val_loss'][-1]  # Last validation loss
#         results.append(val_loss)
    
#     return results

# # Generate learning rates
# learning_rates = np.logspace(-8, -2, num=10)  # Learning rates from 10^-8 to 10^-2

# # Evaluate learning rates
# validation_losses = evaluate_learning_rates(X_train, y_train, learning_rates)

# # Plotting the results
# plt.figure(figsize=(10, 6))
# plt.plot(learning_rates, validation_losses, marker='o')
# plt.xscale('log')
# plt.title('Validation Loss vs Learning Rate')
# plt.xlabel('Learning Rate (log scale)')
# plt.ylabel('Validation Loss')
# plt.grid(True)
# plt.show()

# # Identify the best learning rate
# best_lr = learning_rates[np.argmin(validation_losses)]
# print(f"The best learning rate is: {best_lr:.2e}")




lr_values = np.logspace(-6, -2, num=10)
print(lr_values)




# You can then access the layers just like in a standard Keras model
for layer in best_model.layers:
    print(layer)





for layer in best_model.layers:
    print(f"Layer: {layer.name}, Type: {type(layer)}")





for layer in best_model.layers:
    if isinstance(layer, tf.keras.layers.Bidirectional):
        # Access forward and backward layers
        forward_layer = layer.forward_layer
        backward_layer = layer.backward_layer

        # Print details for the forward layer
        print(f"Bidirectional Layer: {layer.name} (Forward Layer)")
        print(f"  Recurrent Layer Type (Forward): {type(forward_layer).__name__}")
        print(f"  Units (Forward): {forward_layer.units}")
        print(f"  Return Sequences (Forward): {forward_layer.return_sequences}")
        print(f"  Recurrent Dropout (Forward): {forward_layer.recurrent_dropout}")

        # Print details for the backward layer
        print(f"Bidirectional Layer: {layer.name} (Backward Layer)")
        print(f"  Recurrent Layer Type (Backward): {type(backward_layer).__name__}")
        print(f"  Units (Backward): {backward_layer.units}")
        print(f"  Return Sequences (Backward): {backward_layer.return_sequences}")
        print(f"  Recurrent Dropout (Backward): {backward_layer.recurrent_dropout}")
    
    elif isinstance(layer, tf.keras.layers.Dense):
        print(f"Layer: {layer.name}")
        print(f"  Units: {layer.units}")




for layer in best_model.layers:
    if isinstance(layer, tf.keras.layers.LSTM) or isinstance(layer, tf.keras.layers.GRU):
        print(f"Recurrent Layer: {layer.name}")
        print(f"  Units: {layer.units}")
        print(f"  Return Sequences: {layer.return_sequences}")
        print(f"  Recurrent Dropout: {layer.recurrent_dropout}")
    elif isinstance(layer, tf.keras.layers.Dense):
        print(f"Layer: {layer.name}")
        print(f"  Units: {layer.units}")





# Extract optimizer and learning rate
optimizer = best_model.optimizer  # Extract the optimizer
learning_rate = optimizer.learning_rate.numpy()  # Extract the learning rate

print(f"Optimizer: {type(optimizer).__name__}")
print(f"Learning Rate: {learning_rate}")




for layer in best_model.layers:
    layer_config = layer.get_config()  # Get the configuration of each layer
    print(f"Layer: {layer.name}")
    if 'units' in layer_config:
        print(f"  Units: {layer_config['units']}")
    if 'kernel_size' in layer_config:
        print(f"  Kernel Size: {layer_config['kernel_size']}")
    if 'filters' in layer_config:
        print(f"  Number of Filters: {layer_config['filters']}")
    if 'activation' in layer_config:
        print(f"  Activation: {layer_config['activation']}")




for layer in best_model.layers:
    if isinstance(layer, tf.keras.layers.Bidirectional):
        recurrent_layer = layer.forward_layer  # Access the underlying LSTM/GRU layer
        print(f"Bidirectional Layer: {layer.name}")
        print(f"  Recurrent Layer Type (Forward): {type(recurrent_layer).__name__}")
        print(f"  Units (Forward): {recurrent_layer.units}")
        print(f"  Return Sequences (Forward): {recurrent_layer.return_sequences}")
        print(f"  Recurrent Dropout (Forward): {recurrent_layer.recurrent_dropout}")
    elif isinstance(layer, tf.keras.layers.LSTM) or isinstance(layer, tf.keras.layers.GRU):
        print(f"Recurrent Layer: {layer.name}")
        print(f"  Units: {layer.units}")
        print(f"  Return Sequences: {layer.return_sequences}")
        print(f"  Recurrent Dropout: {layer.recurrent_dropout}")




for layer in best_model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        print(f"Dense Layer: {layer.name}")
        print(f"  Units: {layer.units}")
        print(f"  Activation: {layer.activation.__name__}")




for layer in best_model.layers:
    if isinstance(layer, tf.keras.layers.Dropout):
        print(f"Dropout Layer: {layer.name}")
        print(f"  Dropout Rate: {layer.rate}")
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        print(f"Batch Norm Layer: {layer.name}")
    if isinstance(layer, tf.keras.layers.Add):
        print(f"Residual Block: {layer.name}")




pip install pydot




try:
    best_model = model.export_model()
    # Step 8: Visualize the model architecture
    # (a) Print a detailed summary of the architecture
    best_model.summary()
    # (b) Plot the model architecture (save as an image)
    plot_model(best_model, to_file='rnn_model_architecture.png', show_shapes=True)
except IndexError:
    print("No valid model found. Please check the search process and data.")


# #Set Up the Neural Architecture Search



# input_node = ak.Input()  # Use generic Input for various data types
# output_node = ak.DenseBlock()(input_node)  # Explore different dense architectures
# output_node = ak.ClassificationHead()(output_node)  # Define the output layer for classification
# model = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=20, overwrite=True)




# Fit the model
model.fit(X_train,y_train, epochs=50)




predictions=model.predict(X_test)
print(predictions)




# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)




# Get the best performing model after the NAS
best_model = model.export_model()
best_model.summary()  # View the detailed architecture of the best model




# Save the model
best_model.save("best_autokeras_model.h5") 








# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)

# Get predicted probabilities for the positive class
y_pred_prob = model.predict(X_test)

# Check if y_test is binary (1-dimensional array)
if len(y_test.shape) == 1:  
    # y_test is already binary, no need to change
    y_test_binary = y_test
else:
    # For multi-class (e.g., one-hot encoded), select the positive class column
    y_test_binary = y_test[:, 1]

# Calculate AUC-ROC
auc_roc = roc_auc_score(y_test_binary, y_pred_prob)
print("Test AUC-ROC:", auc_roc)

# Optional: Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()




# import pandas as pd

# # Assuming you might want to save more metrics or include more details later
# df = pd.DataFrame({'Metric': ['Accuracy'], 'Value': [accuracy]})
# df.to_csv('model_metrics.csv', index=False)


# # Loading the AutoKeras Model



# # Save the AutoKeras model
# model.export_model().save('model_autokeras_benchmark.h5')





# from tensorflow.keras.models import load_model

# loaded_model = load_model('model_autokeras.h5')




# # Predict on a subset of the test data
# predictions = model.predict(X_test[:20])
# print(predictions)




# from sklearn.metrics import accuracy_score, roc_auc_score
# from scipy.stats import spearmanr
# # Calculate Spearman's Correlation Coefficient
# spearman_corr, _ = spearmanr(y_test, predictions)
# print("Spearman's Correlation Coefficient:", spearman_corr)

