#!/usr/bin/env python
# coding: utf-8

import ipywidgets as widgets
from IPython.display import display
from tkinter import Tk, filedialog
import os
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import autokeras as ak
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, mannwhitneyu
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, roc_curve

#!/usr/bin/env python
# coding: utf-8



pip install ipywidgets








get_ipython().system('pip install autokeras')








# import ipywidgets as widgets
# from IPython.display import display

# # Create a file upload widget
# upload_widget = widgets.FileUpload(accept='', multiple=True)

# # Display the widget
# display(upload_widget)

# # Access uploaded files
# def handle_upload(change):
#     for file_name, file_info in upload_widget.value.items():
#         print(f'Uploaded {file_name} ({file_info["metadata"]["size"]} bytes)')

# # Set the function to run when files are uploaded
# upload_widget.observe(handle_upload, names='value')


# # four cell line datasets




# Paths to your CSV files
file1 = 'hct116.csv'
file2 = 'hela.csv'
file3 = 'hek293.csv'
file4 = 'hl60.csv'

# Read the CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
df4 = pd.read_csv(file4)

# Merge the CSV files by concatenating them
merged_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_output.csv', index=False)

print("CSV files merged successfully!")


# # data balancing



# Count 1s and 0s in each dataset
for i, df in enumerate([df1, df2, df3, df4], start=1):
    print(f"Dataset {i}:")
    print(df['label'].value_counts(), "\n")




for df, name in zip([df1, df2, df3, df4], ["HCT116", "HeLa", "HEK293", "HL60"]):
    label_counts = df["label"].value_counts()
    total = label_counts.sum()
    
    print(f"Dataset: {name}")
    print(f"0s (Negative): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/total:.2%})")
    print(f"1s (Positive): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/total:.2%})")
    print("-" * 40)






for df, name in zip([df1, df2, df3, df4], ["HCT116", "HeLa", "HEK293", "HL60"]):
    plt.figure(figsize=(5, 4))
    sns.countplot(x=df["label"], palette="coolwarm")
    plt.title(f"Label Distribution in {name}")
    plt.xlabel("Label (0 = Negative, 1 = Positive)")
    plt.ylabel("Count")
    plt.show()





datasets = {
    "HCT116": [1521, 2718],
    "HeLa": [3041, 5060],
    "HEK293": [959, 1374],
    "HL60": [602, 1474],
}

df = pd.DataFrame(datasets, index=["Negative (0)", "Positive (1)"]).T
df.plot(kind="bar", stacked=True, figsize=(8, 6), colormap="viridis")

plt.xlabel("Dataset")
plt.ylabel("Count")
plt.title("Class Distribution in Datasets")
plt.xticks(rotation=0)
plt.legend(title="Label")
plt.show()





# Create binary arrays (0s and 1s) from datasets
datasets = {
    "HCT116": np.array([0] * 1521 + [1] * 2718),
    "HeLa": np.array([0] * 3041 + [1] * 5060),
    "HEK293": np.array([0] * 959 + [1] * 1374),
    "HL60": np.array([0] * 602 + [1] * 1474),
}

# Compare label distributions using KS and Mann-Whitney tests
dataset_names = list(datasets.keys())

for i in range(len(dataset_names)):
    for j in range(i + 1, len(dataset_names)):
        ds1, ds2 = dataset_names[i], dataset_names[j]
        
        ks_stat, ks_p = ks_2samp(datasets[ds1], datasets[ds2])
        mw_stat, mw_p = mannwhitneyu(datasets[ds1], datasets[ds2])
        
        print(f"Comparison: {ds1} vs {ds2}")
        print(f" KS Test: D={ks_stat:.4f}, p={ks_p:.4f}")
        print(f" Mann-Whitney U Test: U={mw_stat:.4f}, p={mw_p:.4f}")
        print("-" * 50)




data=pd.read_csv("merged_output.csv")
print(data.head(2))




print(df1.head(2))




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
encoded_df['label'] = data['label']  # Add the efficacy labels back to the DataFrame
encoded_df.head(3)





print(encoded_df.shape)


# # Convert DataFrame to Numpy Array



# X_train = encoded_df.drop('label', axis=1).to_numpy()
# y_train = encoded_df['label'].to_numpy()
X = encoded_df.drop('label', axis=1).to_numpy()
y = encoded_df['label'].to_numpy()




# Step 3: Load the test data (another CSV file)
test_file = 'hl60.csv'  # Replace with the path to your test CSV file
test_df = pd.read_csv(test_file)

# Apply the same one-hot encoding to the test data
encoded_test_data = one_hot_encode_sequences(test_df['sgRNA'])
encoded_test_df = pd.DataFrame(encoded_test_data)
encoded_test_df['label'] = test_df['label']

# Separate features (X) and labels (y) for testing
X_test = encoded_test_df.drop('label', axis=1).to_numpy()
y_test = encoded_test_df['label'].to_numpy()


# # loading pretrained model



# Step 3: Load the pretrained model
best_model = keras.models.load_model("best_autokeras_model.h5")

# Freeze the LSTM layer(s) (assuming the first layer is the LSTM layer)
for layer in best_model.layers[:1]:
    layer.trainable = False

# Check which layers are trainable now
for layer in best_model.layers:
    print(f"Layer: {layer.name}, Trainable: {layer.trainable}")

# Step 4: Recompile the model with a lower learning rate for fine-tuning
optimizer = Adam(learning_rate=1e-5)
best_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Step 5: Fine-tune the model on the merged training data
history = best_model.fit(X_train, y_train, epochs=20, validation_split=0.2)  # No need for train_test_split

# Step 6: Evaluate the model on the separate test file
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")





# Load the pretrained model
best_model = keras.models.load_model("best_autokeras_model.h5")

# Freeze the LSTM layer(s) (optional, if you need fine-tuning later)
for layer in best_model.layers[:1]:
    layer.trainable = False

# Function to one-hot encode sgRNA sequences
def one_hot_encode_sequences(sequences):
    lb = LabelBinarizer()
    lb.fit(list('ACGT'))
    
    encoded_sequences = []
    for seq in sequences:
        encoded_seq = lb.transform(list(seq))  # Encode each nucleotide
        encoded_sequences.append(encoded_seq.flatten())  # Flatten for the model
        
    return np.array(encoded_sequences)

# Load the dataset
data = pd.read_csv("merged_output.csv")

# Original sgRNA sequence (taking the first one as an example)
sgRNA_sequence = data['sgRNA'][0]  # Replace 0 with the index of the sgRNA you want to analyze
sequence_length = len(sgRNA_sequence)

# Define nucleotides
nucleotides = ['A', 'C', 'G', 'T']

# Initialize a matrix to store predictions for each nucleotide substitution
nucleotide_matrix = np.zeros((4, sequence_length))

# Function to modify sequence at a specific position
def modify_sequence(sequence, position, nucleotide):
    modified_seq = list(sequence)
    modified_seq[position] = nucleotide  # Substitute nucleotide at the given position
    return ''.join(modified_seq)

# Loop through each position in the sgRNA sequence
for pos in range(sequence_length):
    # Loop through each nucleotide (A, C, G, T)
    for i, nucleotide in enumerate(nucleotides):
        # Create a modified sequence with the nucleotide at the current position
        modified_sequence = modify_sequence(sgRNA_sequence, pos, nucleotide)
        
        # One-hot encode the modified sequence
        encoded_seq = one_hot_encode_sequences([modified_sequence])
        
        # Get the prediction from the model
        prediction = best_model.predict(encoded_seq)
        
        # Store the prediction in the matrix
        nucleotide_matrix[i, pos] = prediction[0]  # Assuming binary output, take the first value

# Plot the heatmap
plt.figure(figsize=(12, 4))
sns.heatmap(nucleotide_matrix, cmap="YlGnBu", xticklabels=range(1, sequence_length+1), yticklabels=nucleotides)

# Customize the heatmap labels
plt.xlabel('Position in sgRNA Sequence')
plt.ylabel('Nucleotide')
plt.title('Nucleotide Importance Heatmap for sgRNA Sequence')

# Show the heatmap
plt.show()





# Assuming nucleotide_matrix is your data array of size (4, sequence_length)
nucleotide_matrix = np.random.rand(4, 23)  # Example matrix, replace with your actual matrix

# Set up the nucleotide labels for the Y-axis
nucleotides = ['A', 'C', 'G', 'T']

# Plot the heatmap with a similar color scheme to your reference image
plt.figure(figsize=(8, 4))
sns.heatmap(nucleotide_matrix, cmap='RdYlGn', xticklabels=range(1, 24), yticklabels=nucleotides, cbar_kws={'label': 'Importance'})

# Customize the axis labels
plt.xlabel('Position in sgRNA Sequence')
plt.ylabel('Nucleotide')

# Show the plot
plt.show()




# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)  # 70% training, 30% testing





# Load the saved model
best_model = keras.models.load_model("best_autokeras_model.h5")
# best_model = keras.models.load_model("NAS_RNN_dropout.h5")
best_model.summary()


# # Froze Fully connected layer




# Assume your original data is in the variable `X` with shape (num_samples, 92)
# and your target labels are in `y`.

# Define timesteps and features based on your model's input shape
# timesteps = 10
# features = 9

# # Reshape your data to fit your model's input shape
# X = X[:, :timesteps * features].reshape((-1, timesteps, features))
# y = y  # Your target labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the pretrained model
best_model = keras.models.load_model("best_autokeras_model.h5")

# Freeze all layers except the last few
# for layer in best_model.layers[:-2]:  # Freezing all layers except the last 2
#     layer.trainable = False
# Freeze the LSTM layers (analogous to CNN in the pic)
for layer in best_model.layers[:1]:  # Assuming the first layer is the LSTM layer
    layer.trainable = False
# Check which layers are trainable now
for layer in best_model.layers:
    print(f"Layer: {layer.name}, Trainable: {layer.trainable}")

# Recompile the model (important after changing layer trainability)
best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune the model on the new dataset
history = best_model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))




# import tensorflow as tf
# from tensorflow import keras

# # Load the pretrained model
# # best_model = keras.models.load_model("best_autokeras_model.h5")

# # Freeze all layers except the last few
# for layer in best_model.layers[:-2]:  # Freezing all layers except the last 2
#     layer.trainable = False

# # Check which layers are trainable now
# for layer in best_model.layers:
#     print(f"Layer: {layer.name}, Trainable: {layer.trainable}")

# # Recompile the model (important after changing layer trainability)
# best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Fine-tune the model on the new dataset
# history = best_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


# 



# Load the pretrained model
best_model = keras.models.load_model("best_autokeras_model.h5")

# Check which layers are trainable
for layer in best_model.layers:
    print(f"Layer: {layer.name}, Trainable: {layer.trainable}")

# Optionally, freeze some layers (e.g., freeze all but the last layer)
for layer in best_model.layers[:-1]:  # Freezing all layers except the last one
    layer.trainable = False

# Optionally, unfreeze some layers if necessary
# for layer in best_model.layers:  # Unfreezing all layers
#     layer.trainable = True

# Compile the model again after modifying trainability
best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune the model on the new dataset
best_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


# # Fine tunning




# Load the pretrained model
# best_model = keras.models.load_model("NAS_RNN_dropout.h5")

# Initial freeze: keep only the last Dense layers trainable
# for layer in best_model.layers[:-4]:  # Adjust as per your requirement
#     layer.trainable = False
# Freeze all fully connected layers (after the RNN layers)
for layer in best_model.layers[-3:]:  # Adjust based on the model structure
    layer.trainable = False

# Recompile and train the model, focusing on the earlier layers
best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
best_model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test))

# Recompile the model
best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Initial fine-tuning
best_model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test))

# Gradually unfreeze and fine-tune
for layer in best_model.layers[-6:]:  # Adjust to include more layers
    layer.trainable = True

# Recompile the model again
best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Continue fine-tuning
best_model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))

# Repeat the unfreeze and fine-tune process as needed


# # Frozen BLSTM




# Load the pretrained model
best_model = keras.models.load_model("NAS_RNN_dropout.h5")

# Freeze the LSTM (or Bidirectional LSTM) layers specifically
# Assuming the first few layers are LSTM layers, adjust the index if needed
for layer in best_model.layers[:1]:  # Adjust [:1] based on the actual architecture
    if isinstance(layer, keras.layers.LSTM) or isinstance(layer, keras.layers.Bidirectional):
        layer.trainable = False

# Optionally, freeze more layers or adjust which layers you want to keep unfrozen
# For example, freezing all layers except the last fully connected layers
# for layer in best_model.layers[:-2]:  # Freeze all layers except the last 2
#     layer.trainable = False

# Check which layers are now trainable
for layer in best_model.layers:
    print(f"Layer: {layer.name}, Trainable: {layer.trainable}")

# Recompile the model (necessary after changing layer trainability)
best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune the model on the new dataset
history = best_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))








# Evaluate the model on the test set
loss, accuracy = best_model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)

# Get predicted probabilities for the positive class
y_pred_prob = best_model.predict(X_test)

# Handling y_test for AUC-ROC calculation
if len(y_test.shape) == 1:  
    y_test_binary = y_test
else:
    y_test_binary = y_test[:, 1]

# Ensure correct shape for AUC-ROC calculation
if y_pred_prob.shape[1] == 2:
    y_pred_prob_positive_class = y_pred_prob[:, 1]
else:
    y_pred_prob_positive_class = y_pred_prob

# Calculate AUC-ROC
auc_roc = roc_auc_score(y_test_binary, y_pred_prob_positive_class)
print("Test AUC-ROC:", auc_roc)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_prob_positive_class)
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




# print(y_pred_prob)
# Convert probabilities to binary class predictions using a threshold of 0.5
y_pred_class = (y_pred_prob >= 0.5).astype(int)
print(y_pred_class)




y_pred_prob = best_model.predict(X_test)
print(y_pred_prob)
y_pred = np.argmax(y_pred_prob, axis=1)  # If it's a classification problem

