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
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NearMiss, EditedNearestNeighbours
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from collections import Counter
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN
from scipy.stats import pearsonr
import gc
import torch
import autokeras as ak  # AutoKeras is required if the model was trained using it
from sklearn.metrics import accuracy_score, roc_auc_score

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
merged_df = pd.concat([df1, df2, df4], ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_output.csv', index=False)

print("CSV files merged successfully!")




df=pd.read_csv('A549_data.csv')
print(df.head(5))




# Convert log2fc into binary classification
df['label'] = (df['log2fc'] >= 1).astype(int)

# Display class distribution
class_distribution = df['label'].value_counts()
class_distribution




df = df[['sequence', 'label']]




df.to_csv("filtered_A549.csv", index=False)




df.head(3)


# # data balancing



# Count 1s and 0s in each dataset
for i, df in enumerate([df1, df2,df3, df4], start=1):
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
    label_counts = df["label"].value_counts()
    
    majority_class = max(label_counts)
    minority_class = min(label_counts)
    
    imbalance_ratio = majority_class / minority_class
    
    print(f"Dataset: {name}") 
    print(f"Majority Class: {majority_class}")
    print(f"Minority Class: {minority_class}")
    print(f"Imbalance Ratio (IR): {imbalance_ratio:.2f}") 
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




print(df3.head(2))





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
encoded_data = one_hot_encode_sequences(merged_df['sgRNA'])

# Convert the list to a DataFrame for easier handling
encoded_df = pd.DataFrame(encoded_data)
print(encoded_df.head(2))
encoded_df['label'] = merged_df['label']  # Add the efficacy labels back to the DataFrame
encoded_df.head(3)





print(encoded_df.shape)


# # Convert DataFrame to Numpy Array



# X_train = encoded_df.drop('label', axis=1).to_numpy()
# y_train = encoded_df['label'].to_numpy()
X = encoded_df.drop('label', axis=1).to_numpy()
y = encoded_df['label'].to_numpy()




# Step 3: Load the test data (another CSV file)
test_file = 'hek293.csv'  # Replace with the path to your test CSV file
test_df = pd.read_csv(test_file)

# Apply the same one-hot encoding to the test data
encoded_test_data = one_hot_encode_sequences(test_df['sgRNA'])
encoded_test_df = pd.DataFrame(encoded_test_data)
encoded_test_df['label'] = test_df['label']

# Separate features (X) and labels (y) for testing
X_test1 = encoded_test_df.drop('label', axis=1).to_numpy()
y_test1= encoded_test_df['label'].to_numpy()




# import pandas as pd
# from sklearn.preprocessing import LabelBinarizer
# import tensorflow as tf
# from tensorflow import keras
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.optimizers import Adam
# # Step 3: Load the pretrained model
# best_model = keras.models.load_model("best_autokeras_model.h5")

# # Freeze the LSTM layer(s) (assuming the first layer is the LSTM layer)
# for layer in best_model.layers[:1]:
#     layer.trainable = False

# # Check which layers are trainable now
# for layer in best_model.layers:
#     print(f"Layer: {layer.name}, Trainable: {layer.trainable}")

# # Step 4: Recompile the model with a lower learning rate for fine-tuning
# optimizer = Adam(learning_rate=1e-5)
# best_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# # Step 5: Fine-tune the model on the merged training data
# history = best_model.fit(X_train, y_train, epochs=20, validation_split=0.2)  # No need for train_test_split

# # Step 6: Evaluate the model on the separate test file
# test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
# print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")




# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelBinarizer
# from tensorflow import keras

# # Load the pretrained model
# best_model = keras.models.load_model("best_autokeras_model.h5")

# # Freeze the LSTM layer(s) (optional, if you need fine-tuning later)
# for layer in best_model.layers[:1]:
#     layer.trainable = False

# # Function to one-hot encode sgRNA sequences
# def one_hot_encode_sequences(sequences):
#     lb = LabelBinarizer()
#     lb.fit(list('ACGT'))
    
#     encoded_sequences = []
#     for seq in sequences:
#         encoded_seq = lb.transform(list(seq))  # Encode each nucleotide
#         encoded_sequences.append(encoded_seq.flatten())  # Flatten for the model
        
#     return np.array(encoded_sequences)

# # Load the dataset
# data = pd.read_csv("merged_output.csv")

# # Original sgRNA sequence (taking the first one as an example)
# sgRNA_sequence = data['sgRNA'][0]  # Replace 0 with the index of the sgRNA you want to analyze
# sequence_length = len(sgRNA_sequence)

# # Define nucleotides
# nucleotides = ['A', 'C', 'G', 'T']

# # Initialize a matrix to store predictions for each nucleotide substitution
# nucleotide_matrix = np.zeros((4, sequence_length))

# # Function to modify sequence at a specific position
# def modify_sequence(sequence, position, nucleotide):
#     modified_seq = list(sequence)
#     modified_seq[position] = nucleotide  # Substitute nucleotide at the given position
#     return ''.join(modified_seq)

# # Loop through each position in the sgRNA sequence
# for pos in range(sequence_length):
#     # Loop through each nucleotide (A, C, G, T)
#     for i, nucleotide in enumerate(nucleotides):
#         # Create a modified sequence with the nucleotide at the current position
#         modified_sequence = modify_sequence(sgRNA_sequence, pos, nucleotide)
        
#         # One-hot encode the modified sequence
#         encoded_seq = one_hot_encode_sequences([modified_sequence])
        
#         # Get the prediction from the model
#         prediction = best_model.predict(encoded_seq)
        
#         # Store the prediction in the matrix
#         nucleotide_matrix[i, pos] = prediction[0]  # Assuming binary output, take the first value

# # Plot the heatmap
# plt.figure(figsize=(12, 4))
# sns.heatmap(nucleotide_matrix, cmap="YlGnBu", xticklabels=range(1, sequence_length+1), yticklabels=nucleotides)

# # Customize the heatmap labels
# plt.xlabel('Position in sgRNA Sequence')
# plt.ylabel('Nucleotide')
# plt.title('Nucleotide Importance Heatmap for sgRNA Sequence')

# # Show the heatmap
# plt.show()




# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# # Assuming nucleotide_matrix is your data array of size (4, sequence_length)
# nucleotide_matrix = np.random.rand(4, 23)  # Example matrix, replace with your actual matrix

# # Set up the nucleotide labels for the Y-axis
# nucleotides = ['A', 'C', 'G', 'T']

# # Plot the heatmap with a similar color scheme to your reference image
# plt.figure(figsize=(8, 4))
# sns.heatmap(nucleotide_matrix, cmap='RdYlGn', xticklabels=range(1, 24), yticklabels=nucleotides, cbar_kws={'label': 'Importance'})

# # Customize the axis labels
# plt.xlabel('Position in sgRNA Sequence')
# plt.ylabel('Nucleotide')

# # Show the plot
# plt.show()


# # After balancing 




# Paths to your CSV files
file_paths = {
    'HCT116': 'hct116.csv',
    'HeLa': 'hela.csv',
    'HEK293': 'hek293.csv',
    'HL60': 'hl60.csv'
}

# Function to one-hot encode sgRNA sequences
def one_hot_encode_sequences(sequences):
    lb = LabelBinarizer()
    lb.fit(list('ACGT'))  # Encode nucleotides

    encoded_sequences = [lb.transform(list(seq)).flatten() for seq in sequences]
    return np.array(encoded_sequences)

# Dictionary to store processed datasets
datasets = {}

# Load and process each dataset
for dataset_name, file in file_paths.items():
    df = pd.read_csv(file)
    
    # One-hot encode the sgRNA sequences
    encoded_data = one_hot_encode_sequences(df['sgRNA'])
    encoded_df = pd.DataFrame(encoded_data)
    encoded_df['label'] = df['label']  # Add label column
    
    # Split into training and testing sets (80%-20%)
    X = encoded_df.drop('label', axis=1).to_numpy()
    y = encoded_df['label'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Store the split data
    datasets[dataset_name] = {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test
    }

# Define resampling techniques
sampling_methods = {
    "NearMiss": NearMiss(),
    "ENN": EditedNearestNeighbours(),
    "SMOTE": SMOTE(),
    "ADASYN": ADASYN()
}

# Dictionary to store balanced datasets
balanced_datasets = {method: {} for method in sampling_methods}

# Apply sampling techniques to each dataset
for method_name, sampler in sampling_methods.items():
    for dataset_name, data in datasets.items():
        X_resampled, y_resampled = sampler.fit_resample(data['X_train'], data['y_train'])
        balanced_datasets[method_name][dataset_name] = {
            'X': X_resampled, 'y': y_resampled
        }

# Plot the class distribution after balancing
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (method_name, balanced_data) in enumerate(balanced_datasets.items()):
    class_counts = []

    for dataset_name, balanced_df in balanced_data.items():
        unique, counts = np.unique(balanced_df['y'], return_counts=True)
        class_counts.append((dataset_name, counts[0], counts[1]))

    class_df = pd.DataFrame(class_counts, columns=["Dataset", "Negative (0)", "Positive (1)"])
    class_df.set_index("Dataset").plot(kind='bar', stacked=True, ax=axes[i], colormap="viridis")

    axes[i].set_title(f"Class Distribution - {method_name}")
    axes[i].set_xlabel("Dataset")
    axes[i].set_ylabel("Count")

plt.tight_layout()
plt.show()


# # ks and mann whitney test




# Dictionary to store KS and Mann-Whitney U test results
ks_results = {method: {} for method in sampling_methods}
mw_results = {method: {} for method in sampling_methods}

# Perform KS and Mann-Whitney U tests for each dataset and each resampling method
for method_name, balanced_data in balanced_datasets.items():
    for dataset_name, balanced_df in balanced_data.items():
        X_train_original = datasets[dataset_name]['X_train']  # Original Training Data
        X_resampled = balanced_df['X']  # Resampled Data
        
        ks_pvalues = []
        mw_pvalues = []

        # Perform tests on each feature (column-wise comparison)
        for feature_idx in range(X_train_original.shape[1]):
            ks_stat, ks_p = ks_2samp(X_train_original[:, feature_idx], X_resampled[:, feature_idx])
            mw_stat, mw_p = mannwhitneyu(X_train_original[:, feature_idx], X_resampled[:, feature_idx], alternative='two-sided')

            ks_pvalues.append(ks_p)
            mw_pvalues.append(mw_p)

        # Store results as mean p-values across features
        ks_results[method_name][dataset_name] = np.mean(ks_pvalues)
        mw_results[method_name][dataset_name] = np.mean(mw_pvalues)

# Display KS and Mann-Whitney test results
for method in sampling_methods:
    print(f"\n=== KS Test Results for {method} ===")
    for dataset in file_paths.keys():
        avg_ks_p = ks_results[method][dataset]
        print(f"{dataset}: Average p-value = {avg_ks_p:.5f}")

    print(f"\n=== Mann-Whitney U Test Results for {method} ===")
    for dataset in file_paths.keys():
        avg_mw_p = mw_results[method][dataset]
        print(f"{dataset}: Average p-value = {avg_mw_p:.5f}")





# Dictionary to store KS and Mann-Whitney U test results
ks_results = {method: {} for method in sampling_methods}
mw_results = {method: {} for method in sampling_methods}

# Perform KS and Mann-Whitney U tests for each dataset and each resampling method
for method_name, balanced_data in balanced_datasets.items():
    for dataset_name, balanced_df in balanced_data.items():
        X_train_original = datasets[dataset_name]['X_train']  # Original Training Data
        X_resampled = balanced_df['X']  # Resampled Data
        
        ks_pvalues = []
        mw_pvalues = []

        # Perform tests on each feature
        for feature_idx in range(X_train_original.shape[1]):
            ks_stat, ks_p = ks_2samp(X_train_original[:, feature_idx], X_resampled[:, feature_idx])
            mw_stat, mw_p = mannwhitneyu(X_train_original[:, feature_idx], X_resampled[:, feature_idx], alternative='two-sided')

            ks_pvalues.append(ks_p)
            mw_pvalues.append(mw_p)

        # Store results as mean p-values across features
        ks_results[method_name][dataset_name] = np.mean(ks_pvalues)
        mw_results[method_name][dataset_name] = np.mean(mw_pvalues)

# Display KS and Mann-Whitney test results
for method in sampling_methods:
    print(f"\n=== KS Test Results for {method} ===")
    for dataset in file_paths.keys():
        avg_ks_p = ks_results[method][dataset]
        print(f"{dataset}: Average p-value = {avg_ks_p:.5f}")

    print(f"\n=== Mann-Whitney U Test Results for {method} ===")
    for dataset in file_paths.keys():
        avg_mw_p = mw_results[method][dataset]
        print(f"{dataset}: Average p-value = {avg_mw_p:.5f}")

# Plot KDE for one dataset and one method
selected_method = "SMOTE"  # Change this to another method if needed
selected_dataset = "HL60"  # Change to another dataset if needed
X_train_original = datasets[selected_dataset]['X_train']
X_resampled = balanced_datasets[selected_method][selected_dataset]['X']

# Select one feature for visualization (e.g., feature index 0)
feature_idx = 1

plt.figure(figsize=(8, 5))
sns.kdeplot(X_train_original[:, feature_idx], label="Original", color="blue", fill=True, alpha=0.5)
sns.kdeplot(X_resampled[:, feature_idx], label="Resampled", color="red", fill=True, alpha=0.5)

# Add median cut-off lines
plt.axvline(np.median(X_train_original[:, feature_idx]), color="blue", linestyle="--", label="Original Median")
plt.axvline(np.median(X_resampled[:, feature_idx]), color="red", linestyle="--", label="Resampled Median")

# Annotate p-values
ks_p = ks_results[selected_method][selected_dataset]
mw_p = mw_results[selected_method][selected_dataset]
plt.text(0.6, 0.9, f"KS p-value: {ks_p:.3f}\nMW p-value: {mw_p:.3f}",
         transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor="white", alpha=0.5))

plt.title(f"Feature {feature_idx}: Original vs. {selected_method} Resampled")
plt.xlabel("Feature Value")
plt.ylabel("Density")
plt.legend()
plt.show()





# Define the feature index to visualize
feature_idx = 1  # Change this to visualize a different feature

# Define the grid size
num_datasets = len(file_paths)  # Number of datasets
num_methods = len(sampling_methods)  # Number of sampling techniques

fig, axes = plt.subplots(num_datasets, num_methods, figsize=(4 * num_methods, 4 * num_datasets))
fig.suptitle(f"Feature {feature_idx}: Original vs. Resampled KDE Plots", fontsize=16)

# Iterate through datasets and sampling methods
for row_idx, (dataset_name, dataset) in enumerate(datasets.items()):
    for col_idx, (method_name, balanced_data) in enumerate(balanced_datasets.items()):
        ax = axes[row_idx, col_idx] if num_datasets > 1 else axes[col_idx]  # Adjust for single row/col cases

        # Extract original and resampled data
        X_train_original = dataset['X_train']
        X_resampled = balanced_data[dataset_name]['X']

        # Perform statistical tests
        ks_stat, ks_p = ks_2samp(X_train_original[:, feature_idx], X_resampled[:, feature_idx])
        mw_stat, mw_p = mannwhitneyu(X_train_original[:, feature_idx], X_resampled[:, feature_idx], alternative='two-sided')

        # Plot KDE
        sns.kdeplot(X_train_original[:, feature_idx], label="Original", color="blue", fill=True, alpha=0.5, ax=ax)
        sns.kdeplot(X_resampled[:, feature_idx], label=method_name, color="red", fill=True, alpha=0.5, ax=ax)

        # Add median cut-off lines
        ax.axvline(np.median(X_train_original[:, feature_idx]), color="blue", linestyle="--", label="Original Median")
        ax.axvline(np.median(X_resampled[:, feature_idx]), color="red", linestyle="--", label="Resampled Median")

        # Annotate p-values
        ax.text(0.6, 0.8, f"KS p: {ks_p:.3f}\nMW p: {mw_p:.3f}",
                transform=ax.transAxes, fontsize=10, bbox=dict(facecolor="white", alpha=0.5))

        ax.set_title(f"{dataset_name} - {method_name}")
        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Density")
        ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for title
plt.show()




print("Datasets keys:", datasets.keys())
print("Balanced datasets keys:", balanced_datasets.keys())





# Define the feature index to visualize
feature_idx = 1  # Change this to visualize a different feature

# Number of datasets
num_datasets = len(datasets)

# Check available dataset keys and resampling methods
print("Datasets keys:", datasets.keys())
print("Balanced datasets keys:", balanced_datasets.keys())

# Ensure SMOTE is present
if "SMOTE" not in balanced_datasets:
    raise KeyError("SMOTE is missing in balanced_datasets. Check how it was created.")

# Create a single-row figure with multiple subplots (one per dataset)
fig, axes = plt.subplots(1, num_datasets, figsize=(5 * num_datasets, 5))
fig.suptitle(f"Feature {feature_idx}: Original vs. SMOTE KDE Plots", fontsize=16)

# Iterate through datasets and extract corresponding SMOTE data
for idx, (dataset_name, dataset) in enumerate(datasets.items()):
    ax = axes[idx] if num_datasets > 1 else axes  # Handle single dataset case

    # Extract original data
    X_train_original = dataset['X_train']

    # Ensure dataset exists in balanced_datasets['SMOTE']
    if dataset_name in balanced_datasets['SMOTE']:
        X_resampled = balanced_datasets['SMOTE'][dataset_name]['X']
    else:
        print(f"Warning: '{dataset_name}' not found in SMOTE results. Skipping...")
        continue  # Skip this dataset if missing

    # Perform statistical tests
    ks_stat, ks_p = ks_2samp(X_train_original[:, feature_idx], X_resampled[:, feature_idx])
    mw_stat, mw_p = mannwhitneyu(X_train_original[:, feature_idx], X_resampled[:, feature_idx], alternative='two-sided')

    # Plot KDE
    sns.kdeplot(X_train_original[:, feature_idx], label="Original", color="blue", fill=True, alpha=0.5, ax=ax)
    sns.kdeplot(X_resampled[:, feature_idx], label="SMOTE", color="red", fill=True, alpha=0.5, ax=ax)

    # Add median lines
    ax.axvline(np.median(X_train_original[:, feature_idx]), color="blue", linestyle="--", label="Original Median")
    ax.axvline(np.median(X_resampled[:, feature_idx]), color="red", linestyle="--", label="SMOTE Median")

    # Annotate p-values
    ax.text(0.6, 0.8, f"KS p: {ks_p:.3f}\nMW p: {mw_p:.3f}",
            transform=ax.transAxes, fontsize=10, bbox=dict(facecolor="white", alpha=0.5))

    ax.set_title(f"{dataset_name} - SMOTE")
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for title
plt.show()


# # ks and mw test for smote




# Define the feature index to visualize
feature_idx = 1  # Change this to visualize a different feature

# Number of datasets
num_datasets = len(datasets)

# Ensure SMOTE is present
if "SMOTE" not in balanced_datasets:
    raise KeyError("SMOTE is missing in balanced_datasets. Check how it was created.")

# Create a 2x2 figure layout (2 rows, 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
fig.suptitle(f"Feature {feature_idx}: Original vs. SMOTE KDE Plots", fontsize=16)

# Flatten axes for easy iteration
axes = axes.flatten()

# Iterate through datasets and extract corresponding SMOTE data
for idx, (dataset_name, dataset) in enumerate(datasets.items()):
    ax = axes[idx]

    # Extract original data
    X_train_original = dataset['X_train']

    # Ensure dataset exists in balanced_datasets['SMOTE']
    if dataset_name in balanced_datasets['SMOTE']:
        X_resampled = balanced_datasets['SMOTE'][dataset_name]['X']
    else:
        print(f"Warning: '{dataset_name}' not found in SMOTE results. Skipping...")
        continue  # Skip this dataset if missing

    # Perform statistical tests
    ks_stat, ks_p = ks_2samp(X_train_original[:, feature_idx], X_resampled[:, feature_idx])
    mw_stat, mw_p = mannwhitneyu(X_train_original[:, feature_idx], X_resampled[:, feature_idx], alternative='two-sided')

    # Plot KDE
    sns.kdeplot(X_train_original[:, feature_idx], label="Original", color="blue", fill=True, alpha=0.5, ax=ax)
    sns.kdeplot(X_resampled[:, feature_idx], label="SMOTE", color="red", fill=True, alpha=0.5, ax=ax)

    # Add median lines
    ax.axvline(np.median(X_train_original[:, feature_idx]), color="blue", linestyle="--", label="Original Median")
    ax.axvline(np.median(X_resampled[:, feature_idx]), color="red", linestyle="--", label="SMOTE Median")

    # Determine the best text position dynamically
    text_x = np.percentile(X_train_original[:, feature_idx], 98)  # Move text inside denser part
    text_y = ax.get_ylim()[1] * 0.4  # Lower position for better visibility

    # Annotate p-values inside the plot
    ax.annotate(f"KS p: {ks_p:.3f}\nMW p: {mw_p:.3f}",
                xy=(text_x, text_y), fontsize=9, color="black",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"))

    ax.set_title(f"{dataset_name} - SMOTE")
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Density")

    # Move legend outside to avoid overlapping
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, frameon=False)

# Remove any unused subplots (if num_datasets < 4)
for idx in range(num_datasets, 4):
    fig.delaxes(axes[idx])

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for title
plt.savefig('KS and mw test.png')
plt.show()





# Select a feature index (e.g., 0 for the first feature)
feature_index = 0  

# Extract the same feature from original and resampled datasets
original_feature = X_train_original[:, feature_index]  # If using NumPy
resampled_feature = X_train_resampled[:, feature_index]  # If using NumPy

# OR if using Pandas DataFrame
# original_feature = X_train_original.iloc[:, feature_index]MMMMMMMMMMMM_
# resampled_feature = X_train_resampled.iloc[:, feature_index]

# Plot the histogram
plt.hist(original_feature, bins=10, alpha=0.5, label="Original")
plt.hist(resampled_feature, bins=10, alpha=0.5, label="Resampled")
plt.legend()
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.title(f"Histogram of Feature {feature_index}")
plt.show()




print("Unique values in Feature 0:", np.unique(X_train[:, 0]))





plt.hist(X_train_original[:, 0], bins=50, alpha=0.7, label='Original')
plt.hist(X_resampled[:, 0], bins=50, alpha=0.7, label='SMOTE Resampled')
plt.legend()
plt.xlabel('Feature 0 Value')
plt.ylabel('Frequency')
plt.title('Histogram of Feature 0')
plt.show()




# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)  # 70% training, 30% testing





# Initialize undersampler
undersampler = RandomUnderSampler(random_state=42)

# Apply undersampling on the training data
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

# Print the class distribution after downsampling
unique, counts = np.unique(y_train_resampled, return_counts=True)
print("Class distribution after downsampling:", dict(zip(unique, counts)))






# Define NearMiss object
nearmiss = NearMiss(version=1)  # You can change the version (1, 2, or 3) based on your needs

# Apply NearMiss to training data
X_train_resampled, y_train_resampled = nearmiss.fit_resample(X_train, y_train)

# Print new class distribution
print("Class distribution after NearMiss undersampling:", Counter(y_train_resampled))





# Define ENN with k=3 neighbors
enn = EditedNearestNeighbours(n_neighbors=3)

# Apply ENN
X_train_resampled, y_train_resampled = enn.fit_resample(X_train, y_train)

print("Class distribution after ENN:", Counter(y_train_resampled))





# Define SMOTE object
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Apply SMOTE to training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Print new class distribution
print("Class distribution after oversampling:", Counter(y_train_resampled))




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
history = best_model.fit(X_train_resampled, y_train_resampled, epochs=20, validation_split=0.2)  # No need for train_test_split

# Step 6: Evaluate the model on the separate test file
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")





# Define RandomOverSampler object
ros = RandomOverSampler(sampling_strategy='auto', random_state=42)

# Apply random oversampling to training data
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Print new class distribution
print("Class distribution after random oversampling:", Counter(y_train_resampled))





# Define ADASYN object
adasyn = ADASYN(sampling_strategy='auto', random_state=42)

# Apply ADASYN to training data
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)

# Print new class distribution
print("Class distribution after ADASYN oversampling:", Counter(y_train_resampled))





# Load the saved model
best_model = keras.models.load_model("best_autokeras_model.h5")
# best_model = keras.models.load_model("NAS_RNN_dropout.h5")
best_model.summary()




print(y_pred)
print(y_test)
print(X_test[0])




pearson_r, p_value = pearsonr(y_test, y_pred)
print(f"Pearson Correlation: {pearson_r}, P-value: {p_value}")





# Free GPU memory before loading the model
torch.cuda.empty_cache()
gc.collect()

# Enable memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Define the missing custom layer
class CastToFloat32(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

# Register all necessary custom objects
custom_objects = {
    "Dropout": tf.keras.layers.Dropout,
    "Custom>CastToFloat32": CastToFloat32,  # AutoKeras naming
    "cast_to_float32": CastToFloat32,  # Standard TensorFlow naming
}

# Load the pretrained model safely
try:
    best_model = tf.keras.models.load_model("best_autokeras_model.h5", custom_objects=custom_objects, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Ensure input shape matches expected
expected_input_shape = best_model.input_shape[1:]
if X_test.shape[1:] != expected_input_shape:
    raise ValueError(f"❌ Shape Mismatch! X_test shape: {X_test.shape}, expected: {expected_input_shape}")

# Prevent training on BatchNorm layers
for layer in best_model.layers[-5:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

# Compile with a smaller learning rate for fine-tuning
best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='mse', metrics=['mae'])





# ------------------------------ #
# 1️⃣ Load & Preprocess Data     #
# ------------------------------ #

# Function to one-hot encode sgRNA sequences
def one_hot_encode_sequences(sequences):
    lb = LabelBinarizer()
    lb.fit(list("ACGT"))  # Encode nucleotides A, C, G, T
    encoded_sequences = [lb.transform(list(seq)).flatten() for seq in sequences]  # Flatten encoding
    return np.array(encoded_sequences, dtype=np.float32)

# Load independent test dataset (A549) and keep only 'sequence' and 'log2fc' columns
df_test = pd.read_csv("A549_data.csv", usecols=["sequence", "log2fc"])  # ✅ Ensure A549 dataset is used

# One-hot encode sequences
X_test = one_hot_encode_sequences(df_test['sequence'])

# Print shape for verification
print("✅ Model input shape:", X_test.shape)

# Normalize target
y_test_raw = df_test["log2fc"].values
y_min, y_max = y_test_raw.min(), y_test_raw.max()
y_test = (y_test_raw - y_min) / (y_max - y_min)

# ------------------------------ #
# 2️⃣ Load & Modify Pretrained Model #
# ------------------------------ #

# ✅ Handle 'Custom>CastToFloat32' Layer While Loading
try:
    best_model = keras.models.load_model(
        "best_autokeras_model.h5",
        custom_objects={'Custom>CastToFloat32': tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))}
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Verify input shape
expected_input_shape = best_model.input_shape[1:]
print(f"🔹 Expected Input Shape: {expected_input_shape}")

# ✅ Auto-reshape X_test if needed
if X_test.shape[1:] != expected_input_shape:
    print(f"⚠️ Shape Mismatch! Reshaping X_test from {X_test.shape} to match model input {expected_input_shape}...")
    X_test = X_test.reshape(-1, *expected_input_shape)

# Modify last layer for regression if needed
if isinstance(best_model.layers[-1], keras.layers.Dense) and (
    best_model.layers[-1].units != 1 or best_model.layers[-1].activation != keras.activations.linear
):
    print("🔹 Modifying last layer for regression...")
    base_model = keras.Model(inputs=best_model.input, outputs=best_model.layers[-2].output)
    output = keras.layers.Dense(1, activation='linear')(base_model.output)
    best_model = keras.Model(inputs=base_model.input, outputs=output)

    # Compile the model for regression
    optimizer = Adam(learning_rate=1e-5)
    best_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# ------------------------------ #
# 3️⃣ Check Data Distribution & Domain Shift #
# ------------------------------ #

plt.figure(figsize=(8, 5))
plt.hist(y_test_raw, bins=50, alpha=0.5, label="A549 Data")
plt.xlabel("log2fc Value")
plt.ylabel("Frequency")
plt.title("Distribution of log2fc Values (A549)")
plt.legend()
plt.show()

# ------------------------------ #
# 4️⃣ Predict & Debug Issues    #
# ------------------------------ #

# Predict log2fc values on the independent A549 dataset
y_pred_raw = best_model.predict(X_test).flatten()

# Print statistics for debugging
print("\n🔹 Prediction Statistics:")
print(f"Mean Prediction: {np.mean(y_pred_raw)}")
print(f"Std Dev of Predictions: {np.std(y_pred_raw)}")
print(f"Min Prediction: {np.min(y_pred_raw)}, Max Prediction: {np.max(y_pred_raw)}")

# ------------------------------ #
# 5️⃣ Compute Pearson Correlation #
# ------------------------------ #

# Compute Pearson correlation coefficient
pearson_corr, p_value = pearsonr(y_test, y_pred_raw)
print(f"\n✅ Pearson Correlation Coefficient: {pearson_corr:.4f}, P-value: {p_value:.4e}")

# ------------------------------ #
# 6️⃣ Fine-Tune Model on A549 Data #
# ------------------------------ #

# Split A549 data into training & validation for fine-tuning
X_train, X_val, y_train, y_val = train_test_split(X_test, y_test_raw, test_size=0.2, random_state=42)

# ✅ Unfreeze only trainable layers (skip BatchNorm layers)
for layer in best_model.layers[-5:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

# Recompile model with a smaller learning rate
best_model.compile(optimizer=Adam(learning_rate=1e-5), loss='mse', metrics=['mae'])

# Train for a few epochs
print("\n🔹 Fine-Tuning on A549 Data...")
history = best_model.fit(X_train, y_train, epochs=80, validation_data=(X_val, y_val), batch_size=32)

# Evaluate on full A549 test set after fine-tuning
y_pred_finetuned = best_model.predict(X_test).flatten()

# Compute Pearson correlation after fine-tuning
pearson_corr_finetuned, p_value_finetuned = pearsonr(y_test, y_pred_finetuned)
print(f"\n✅ Pearson Correlation After Fine-Tuning: {pearson_corr_finetuned:.4f}, P-value: {p_value_finetuned:.4e}")


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




# # Load the pretrained model
# best_model = keras.models.load_model("best_autokeras_model.h5")

# # Check which layers are trainable
# for layer in best_model.layers:
#     print(f"Layer: {layer.name}, Trainable: {layer.trainable}")

# # Optionally, freeze some layers (e.g., freeze all but the last layer)
# for layer in best_model.layers[:-1]:  # Freezing all layers except the last one
#     layer.trainable = False

# # Optionally, unfreeze some layers if necessary
# # for layer in best_model.layers:  # Unfreezing all layers
# #     layer.trainable = True

# # Compile the model again after modifying trainability
# best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Fine-tune the model on the new dataset
# best_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


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
best_model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Repeat the unfreeze and fine-tune process as needed





# Get predictions (probabilities)
y_pred_probs = best_model.predict(X_test)

# Convert probabilities to binary predictions (0 or 1) using 0.5 threshold
y_pred = (y_pred_probs >= 0.5).astype(int)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Compute AUC-ROC
auc_roc = roc_auc_score(y_test, y_pred_probs)
print(f"Test AUC-ROC: {auc_roc:.4f}")




print(y_pred)

