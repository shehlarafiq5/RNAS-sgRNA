import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import autokeras as ak
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import h5py
import autokeras as ak  # AutoKeras might be needed if the model was trained with it
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
from collections import Counter
import shap
import seaborn as sns
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install autokeras')








df=pd.read_csv('A549_data.csv')
print(df.head(5))




# Convert log2fc into binary classification
df['label'] = (df['log2fc'] >= 1).astype(int)

# Display class distribution
class_distribution = df['label'].value_counts()
class_distribution




df = df[['sequence', 'label']]




df.to_csv("filtered_A549.csv", index=False)





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
encoded_data = one_hot_encode_sequences(df['sequence'])

# Convert the list to a DataFrame for easier handling
encoded_df = pd.DataFrame(encoded_data)
print(encoded_df.head(2))
encoded_df['label'] = df['label']  # Add the efficacy labels back to the DataFrame
encoded_df.head(3)





# X_train = encoded_df.drop('label', axis=1).to_numpy()
# y_train = encoded_df['label'].to_numpy()
X = encoded_df.drop('label', axis=1).to_numpy()
y = encoded_df['label'].to_numpy()





# Normalize the target labels (y) for regression
scaler = MinMaxScaler()
y = scaler.fit_transform(y.reshape(-1, 1)).flatten()  # Normalize y between 0 and 1

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the pretrained model
best_model = keras.models.load_model("best_autokeras_model.h5")

# Identify the last Dense layer before activation
for layer in reversed(best_model.layers):
    if isinstance(layer, keras.layers.Dense):
        last_dense_layer = layer
        break

# Modify the last layer if it's not suitable for regression
if last_dense_layer.units != 1 or last_dense_layer.activation != keras.activations.linear:
    new_output = keras.layers.Dense(1, activation="linear", name="regression_head")(last_dense_layer.input)
    best_model = keras.Model(inputs=best_model.input, outputs=new_output)

# Freeze the first layer (assumed to be feature extraction)
for layer in best_model.layers[:1]:  
    layer.trainable = False

# Print trainable layers
for layer in best_model.layers:
    print(f"Layer: {layer.name}, Trainable: {layer.trainable}")

# Compile the model for regression
best_model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # MSE for regression

# Fine-tune the model
history = best_model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test))

# Get model predictions
y_pred = best_model.predict(X_test).flatten()  # Ensure predictions are 1D

# Compute Pearson correlation coefficient and p-value
pearson_coeff, p_value = pearsonr(y_test, y_pred)

print(f"Pearson Correlation Coefficient: {pearson_coeff:.4f}")
print(f"P-value: {p_value:.4e}")  # Scientific notation for readability





# Open the model file and check layer names
model_path = "best_autokeras_model.h5"
with h5py.File(model_path, "r") as f:
    print("Model Layers:", list(f["model_weights"].keys()))





# Define a wrapper to register the missing custom layer
class CastToFloat32(keras.layers.Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

# Map both possible names
custom_objects = {
    "Custom>CastToFloat32": CastToFloat32,  # AutoKeras may save it with "Custom>" prefix
    "cast_to_float32": CastToFloat32,  # Standard TensorFlow naming
}

# Path to your model
model_path = "best_autokeras_model.h5"

try:
    # Load the model with both possible custom layer names
    best_model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects, 
        compile=False
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Print model summary
best_model.summary()


# # Independent datasets for generalisation




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
# df_test = pd.read_csv("K562_data.csv", usecols=["sequence", "log2fc"])
df_test = pd.read_csv("A549_data.csv", usecols=["sequence", "log2fc"])
# df_test = pd.read_csv("NB4_data.csv", usecols=["sequence", "log2fc"])

# Ensure sequence column is properly one-hot encoded
X_test = one_hot_encode_sequences(df_test['sequence'])

# Print shapes to verify compatibility
print("✅ Model input shape:", X_test.shape)

# Extract and normalize target (log2fc)
y_test_raw = df_test["log2fc"].values

# Store original min-max for later debugging
y_min, y_max = y_test_raw.min(), y_test_raw.max()
print(f"🔹 y_min: {y_min}, y_max: {y_max}")

# Normalize target
y_test = (y_test_raw - y_min) / (y_max - y_min)

# ------------------------------ #
# 2️⃣ Load & Modify Pretrained Model #
# ------------------------------ #

# Load pretrained model
best_model = keras.models.load_model("best_autokeras_model.h5")

# Print model summary to check input layer shape
best_model.summary()

# Verify input shape
expected_input_shape = best_model.input_shape[1:]
print(f"🔹 Expected Input Shape: {expected_input_shape}")

# Check if X_test matches expected input shape
if X_test.shape[1:] != expected_input_shape:
    raise ValueError(f"❌ Shape Mismatch! X_test shape: {X_test.shape}, expected: {expected_input_shape}")

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

# Compare distribution of training and A549 data
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

# Normalize predictions using same min-max
y_pred = y_pred_raw  # Do NOT apply min-max normalization

# Print statistics for debugging
print("\n🔹 Prediction Statistics:")
print(f"Mean Prediction: {np.mean(y_pred_raw)}")
print(f"Std Dev of Predictions: {np.std(y_pred_raw)}")
print(f"Min Prediction: {np.min(y_pred_raw)}, Max Prediction: {np.max(y_pred_raw)}")

# ------------------------------ #
# 5️⃣ Compute Pearson Correlation #
# ------------------------------ #

# Compute Pearson correlation coefficient
pearson_corr, p_value = pearsonr(y_test, y_pred)
print(f"\n✅ Pearson Correlation Coefficient: {pearson_corr:.4f}, P-value: {p_value:.4e}")

# ------------------------------ #
# 6️⃣ Fine-Tune Model on A549 Data #
# ------------------------------ #


# Split A549 data into training & validation for fine-tuning
X_train, X_val, y_train, y_val = train_test_split(X_test, y_test_raw, test_size=0.2, random_state=42)

# Unfreeze last few layers for fine-tuning
for layer in best_model.layers[-5:]:  # Fine-tune last 5 layers
    layer.trainable = True

# Recompile model with a smaller learning rate
best_model.compile(optimizer=Adam(learning_rate=1e-5), loss='mse', metrics=['mae'])

# Train for a few epochs
print("\n🔹 Fine-Tuning on A549 Data...")
history = best_model.fit(X_train, y_train, epochs=35, validation_data=(X_val, y_val), batch_size=32)

# Evaluate on full A549 test set after fine-tuning
y_pred_finetuned = best_model.predict(X_test).flatten()

# Compute Pearson correlation after fine-tuning
pearson_corr_finetuned, p_value_finetuned = pearsonr(y_test, y_pred_finetuned)
print(f"\n✅ Pearson Correlation After Fine-Tuning: {pearson_corr_finetuned:.4f}, P-value: {p_value_finetuned:.4e}")


# # 1 LIME AND SHAP




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




merged_df.head(2)





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



# # for shap values  




# Function to one-hot encode sgRNA sequences
def one_hot_encode_sequences(sequences):
    lb = LabelBinarizer()
    lb.fit(list('ACGT'))  # Fit only on A, C, G, T

    encoded_sequences = []
    for seq in sequences:
        encoded_seq = lb.transform(list(seq))  # One-hot encode each nucleotide
        encoded_sequences.append(encoded_seq.flatten())  # Flatten to 1D

    return encoded_sequences

# Apply encoding
encoded_data = one_hot_encode_sequences(merged_df['sgRNA'])

# Generate feature names based on position and nucleotide
sequence_length = len(merged_df['sgRNA'][0])  # Assuming all sequences have the same length
nucleotides = ['A', 'C', 'G', 'T']
feature_names = [f"Pos_{i}_{nuc}" for i in range(sequence_length) for nuc in nucleotides]

# Convert to DataFrame with proper column names
encoded_df = pd.DataFrame(encoded_data, columns=feature_names)
encoded_df['label'] = merged_df['label']  # Add label column

print(encoded_df.head(3))




# X_train = encoded_df.drop('label', axis=1).to_numpy()
# y_train = encoded_df['label'].to_numpy()
X = encoded_df.drop('label', axis=1).to_numpy()
y = encoded_df['label'].to_numpy()




# Instead of converting to NumPy, keep X as a DataFrame
X = encoded_df.drop(columns=['label'])  # Keep as DataFrame
y = encoded_df['label'].to_numpy()  # y can remain a NumPy array




# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)  # 70% training, 30% testing





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




pip install shap





# Initialize the SHAP explainer
explainer = shap.Explainer(best_model, X_train_resampled[:100])  # Use a small sample for efficiency

# Compute SHAP values
shap_values = explainer(X_test[:50])  # Explain predictions for a subset of test data




print(type(shap_values))  # Should be <class 'shap.Explanation'>
print(shap_values.shape)  # Expected: (50, 92), but likely (50, 1)




print(best_model.output_shape)  # Should be (None, 92) for feature attributions





# Ensure X_train_resampled and X_test are in float32 format
X_train_resampled = X_train_resampled.astype(np.float32)
X_test = X_test.astype(np.float32)

# Ensure the model's input is float32 using a TensorFlow cast layer
input_layer = best_model.input
if input_layer.dtype != tf.float32:
    input_layer = tf.keras.layers.Input(shape=input_layer.shape[1:], dtype=tf.float32)

# Recompile the model if necessary (not usually needed)
best_model.compile()

# Use SHAP Explainer
explainer = shap.Explainer(best_model, X_train_resampled[:100])

# Compute SHAP values
shap_values = explainer(X_test[:50])

# Convert SHAP values to NumPy array if needed
shap_values_array = shap_values.values if hasattr(shap_values, "values") else np.array(shap_values)

# Verify output
print(f"SHAP values type: {type(shap_values_array)}")  
print(f"SHAP values shape: {shap_values_array.shape}")  # Should match (50, num_features)

# Visualize results
shap.summary_plot(shap_values, X_test[:50])





# Convert SHAP values to a NumPy array if needed
shap_values_array = np.array(shap_values.values)

# Create a heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(shap_values_array, cmap='coolwarm', center=0, annot=False)
plt.xlabel("Features (Nucleotide Positions)")
plt.ylabel("sgRNA Sequence")
plt.title("SHAP Value Heatmap")
plt.show()


# # final shap heat map




# Convert SHAP values to NumPy array
shap_values_array = np.array(shap_values.values)  # Actual SHAP values

# Check the shape
print("Original SHAP shape:", shap_values_array.shape)

# Extract dimensions
num_samples, num_features = shap_values_array.shape
sequence_length = num_features // 4  # Since each position has 4 nucleotides

# Reshape SHAP values: Convert (num_samples, sequence_length × 4) → (num_samples, sequence_length, 4)
shap_values_reshaped = shap_values_array.reshape(num_samples, sequence_length, 4)

# Take mean across all samples → Shape becomes (4, sequence_length)
shap_mean = shap_values_reshaped.mean(axis=0).T  

# Define nucleotide labels
nucleotides = ['A', 'C', 'G', 'T']

# Create heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(shap_mean, cmap='coolwarm', center=0, 
            xticklabels=range(1, sequence_length + 1), 
            yticklabels=nucleotides, 
            cbar_kws={'label': 'SHAP Importance'})

# Labels and title
plt.xlabel("Position in sgRNA Sequence")
plt.ylabel("Nucleotide")
plt.title("SHAP Value Heatmap for sgRNA Positions")

# Show plot
plt.show()




print("SHAP shape:", shap_values.values.shape)
print("X shape:", X.shape)





# Compute mean absolute SHAP values for feature sorting
mean_shap_values = np.abs(shap_values.values).mean(axis=0)

# Sort feature indices by importance
sorted_indices = np.argsort(-mean_shap_values)

# Convert X_test to NumPy before indexing
X_test_np = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test

# Sort SHAP values and test dataset accordingly
shap_values_sorted = shap_values.values[:, sorted_indices]
X_test_sorted = X_test_np[:, sorted_indices]  # Now works correctly

# Create the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(
    shap_values_sorted, 
    cmap="coolwarm", 
    center=0, 
    xticklabels=sorted_indices, 
    yticklabels=False,  # Hide individual sgRNA labels
    cbar_kws={'label': 'SHAP Value'}
)

# Axis labels
plt.xlabel("Features (Sorted Nucleotide Positions)")
plt.ylabel("sgRNA Sequences")
plt.title("SHAP Value Heatmap (Sorted Features)")
plt.show()




print("X_test shape:", X_test.shape)  # Should be (num_samples, num_features)
print("shap_values shape:", shap_values.shape)  # Should match X_test

# Verify available feature indices
num_features = X_test.shape[1]
print("Number of features:", num_features)




shap_values = np.array(shap_values)




print(type(shap_values))
print([sv.shape for sv in shap_values])  # Check shapes




print(shap_values.shape)  # Should ideally be (50, 92)


# # GC COUNT formula-  (Number of G + Number of C) / Total Length of Sequence * 100% 




# Function to compute GC content
def compute_gc_content(sequence):
    return (sequence.count('G') + sequence.count('C')) / len(sequence)

# Function to one-hot encode sgRNA sequences
def one_hot_encode_sequences(sequences):
    lb = LabelBinarizer()
    lb.fit(list('ACGT'))
    encoded_sequences = [lb.transform(list(seq)).flatten() for seq in sequences]
    return np.array(encoded_sequences)

# Function to rebuild model with new input shape
def rebuild_model_with_new_input(model, new_input_dim):
    new_input = Input(shape=(new_input_dim,), name="new_input")
    x = new_input

    for layer in model.layers[1:]:
        if isinstance(layer, Dense):
            x = Dense(units=layer.units, activation=layer.activation)(x)
        else:
            x = layer(x)

    new_model = Model(inputs=new_input, outputs=x, name="updated_model")

    for old_layer, new_layer in zip(model.layers[1:], new_model.layers[1:]):
        if isinstance(old_layer, Dense):
            old_weights = old_layer.get_weights()
            if old_weights:  # Check if weights exist
                w, b = old_weights
                if w.shape[0] + 1 == new_input_dim:  # Match new input shape
                    w = np.vstack([w, np.random.normal(scale=0.01, size=(1, w.shape[1]))])
                new_layer.set_weights([w, b])

    return new_model

# Load cell line datasets
cell_lines = {
    'HCT116': 'hct116.csv',
    'HeLa': 'hela.csv',
    'HEK293': 'hek293.csv',
    'HL60': 'hl60.csv'
}

results = {}

for name, file in cell_lines.items():
    print(f"\nProcessing {name} cell line...")
    
    df = pd.read_csv(file)
    df['GC_Content'] = df['sgRNA'].apply(compute_gc_content)
    
    X_encoded = one_hot_encode_sequences(df['sgRNA'])
    X_with_gc = np.hstack([X_encoded, df['GC_Content'].values.reshape(-1, 1)])
    
    y = df['label'].to_numpy()
    
    X_train_no_gc, X_test_no_gc, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    X_train_with_gc, X_test_with_gc, _, _ = train_test_split(X_with_gc, y, test_size=0.2, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_resampled_no_gc, y_train_resampled = smote.fit_resample(X_train_no_gc, y_train)
    X_train_resampled_with_gc, _ = smote.fit_resample(X_train_with_gc, y_train)
    
    best_model = keras.models.load_model("best_autokeras_model.h5")
    
    for layer in best_model.layers[:1]:
        layer.trainable = False
    
    # Clear session before modifying the model
    K.clear_session()

    # Compile best_model with a new optimizer
    best_model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    history_no_gc = best_model.fit(X_train_resampled_no_gc, y_train_resampled, epochs=40, validation_split=0.2, verbose=0)
    
    model_with_gc = rebuild_model_with_new_input(best_model, new_input_dim=93)
    
    # Compile model_with_gc with a new optimizer to avoid variable mismatch
    model_with_gc.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    history_with_gc = model_with_gc.fit(X_train_resampled_with_gc, y_train_resampled, epochs=20, validation_split=0.2, verbose=0)
    
    y_pred_no_gc = best_model.predict(X_test_no_gc).flatten()
    y_pred_with_gc = model_with_gc.predict(X_test_with_gc).flatten()
    
    acc_no_gc = accuracy_score(y_test, np.round(y_pred_no_gc))
    auc_no_gc = roc_auc_score(y_test, y_pred_no_gc)
    
    acc_with_gc = accuracy_score(y_test, np.round(y_pred_with_gc))
    auc_with_gc = roc_auc_score(y_test, y_pred_with_gc)
    
    results[name] = {
        "Accuracy Without GC": acc_no_gc,
        "AUROC Without GC": auc_no_gc,
        "Accuracy With GC": acc_with_gc,
        "AUROC With GC": auc_with_gc
    }

print("\nPerformance Comparison (With vs Without GC Count):")
for cell, metrics in results.items():
    print(f"\n{cell} Cell Line:")
    print(f"  Accuracy Without GC: {metrics['Accuracy Without GC']:.4f}")
    print(f"  AUROC Without GC: {metrics['AUROC Without GC']:.4f}")
    print(f"  Accuracy With GC: {metrics['Accuracy With GC']:.4f}")
    print(f"  AUROC With GC: {metrics['AUROC With GC']:.4f}")





# Convert dictionary to DataFrame
results_df = pd.DataFrame.from_dict(results, orient="index")

# Save to CSV
results_df.to_csv("gc_content_results.csv")

print("\nPerformance metrics saved to 'gc_content_results.csv'")





# Adding precision, recall, and F1 score calculation to the results

for name, file in cell_lines.items():
    print(f"\nProcessing {name} cell line...")

    df = pd.read_csv(file)
    df['GC_Content'] = df['sgRNA'].apply(compute_gc_content)

    X_encoded = one_hot_encode_sequences(df['sgRNA'])
    X_with_gc = np.hstack([X_encoded, df['GC_Content'].values.reshape(-1, 1)])

    y = df['label'].to_numpy()

    X_train_no_gc, X_test_no_gc, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    X_train_with_gc, X_test_with_gc, _, _ = train_test_split(X_with_gc, y, test_size=0.2, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_resampled_no_gc, y_train_resampled = smote.fit_resample(X_train_no_gc, y_train)
    X_train_resampled_with_gc, _ = smote.fit_resample(X_train_with_gc, y_train)

    best_model = keras.models.load_model("best_autokeras_model.h5")

    for layer in best_model.layers[:1]:
        layer.trainable = False

    # Clear session before modifying the model
    K.clear_session()

    # Compile best_model with a new optimizer
    best_model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    history_no_gc = best_model.fit(X_train_resampled_no_gc, y_train_resampled, epochs=40, validation_split=0.2, verbose=0)

    model_with_gc = rebuild_model_with_new_input(best_model, new_input_dim=93)

    # Compile model_with_gc with a new optimizer to avoid variable mismatch
    model_with_gc.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    history_with_gc = model_with_gc.fit(X_train_resampled_with_gc, y_train_resampled, epochs=20, validation_split=0.2, verbose=0)

    y_pred_no_gc = best_model.predict(X_test_no_gc).flatten()
    y_pred_with_gc = model_with_gc.predict(X_test_with_gc).flatten()

    acc_no_gc = accuracy_score(y_test, np.round(y_pred_no_gc))
    auc_no_gc = roc_auc_score(y_test, y_pred_no_gc)
    precision_no_gc = precision_score(y_test, np.round(y_pred_no_gc))
    recall_no_gc = recall_score(y_test, np.round(y_pred_no_gc))
    f1_no_gc = f1_score(y_test, np.round(y_pred_no_gc))

    acc_with_gc = accuracy_score(y_test, np.round(y_pred_with_gc))
    auc_with_gc = roc_auc_score(y_test, y_pred_with_gc)
    precision_with_gc = precision_score(y_test, np.round(y_pred_with_gc))
    recall_with_gc = recall_score(y_test, np.round(y_pred_with_gc))
    f1_with_gc = f1_score(y_test, np.round(y_pred_with_gc))

    results[name] = {
        "Accuracy Without GC": acc_no_gc,
        "AUROC Without GC": auc_no_gc,
        "Precision Without GC": precision_no_gc,
        "Recall Without GC": recall_no_gc,
        "F1 Score Without GC": f1_no_gc,
        "Accuracy With GC": acc_with_gc,
        "AUROC With GC": auc_with_gc,
        "Precision With GC": precision_with_gc,
        "Recall With GC": recall_with_gc,
        "F1 Score With GC": f1_with_gc
    }

print("\nPerformance Comparison (With vs Without GC Count):")
for cell, metrics in results.items():
    print(f"\n{cell} Cell Line:")
    print(f"  Accuracy Without GC: {metrics['Accuracy Without GC']:.4f}")
    print(f"  AUROC Without GC: {metrics['AUROC Without GC']:.4f}")
    print(f"  Precision Without GC: {metrics['Precision Without GC']:.4f}")
    print(f"  Recall Without GC: {metrics['Recall Without GC']:.4f}")
    print(f"  F1 Score Without GC: {metrics['F1 Score Without GC']:.4f}")
    print(f"  Accuracy With GC: {metrics['Accuracy With GC']:.4f}")
    print(f"  AUROC With GC: {metrics['AUROC With GC']:.4f}")
    print(f"  Precision With GC: {metrics['Precision With GC']:.4f}")
    print(f"  Recall With GC: {metrics['Recall With GC']:.4f}")
    print(f"  F1 Score With GC: {metrics['F1 Score With GC']:.4f}")





# Convert results to DataFrame
results_df = pd.DataFrame(results).T  # Transpose the results dictionary to make cell lines rows

# Save the DataFrame to a CSV file
results_df.to_csv("performance_comparison_results(f1 score).csv", index=True)

print("\nResults saved to 'performance_comparison_results.csv'")


# # All performance metrics




# Adding specificity calculation to the results

for name, file in cell_lines.items():
    print(f"\nProcessing {name} cell line...")

    df = pd.read_csv(file)
    df['GC_Content'] = df['sgRNA'].apply(compute_gc_content)

    X_encoded = one_hot_encode_sequences(df['sgRNA'])
    X_with_gc = np.hstack([X_encoded, df['GC_Content'].values.reshape(-1, 1)])

    y = df['label'].to_numpy()

    X_train_no_gc, X_test_no_gc, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    X_train_with_gc, X_test_with_gc, _, _ = train_test_split(X_with_gc, y, test_size=0.2, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_resampled_no_gc, y_train_resampled = smote.fit_resample(X_train_no_gc, y_train)
    X_train_resampled_with_gc, _ = smote.fit_resample(X_train_with_gc, y_train)

    best_model = keras.models.load_model("best_autokeras_model.h5")

    for layer in best_model.layers[:1]:
        layer.trainable = False

    # Clear session before modifying the model
    K.clear_session()

    # Compile best_model with a new optimizer
    best_model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    history_no_gc = best_model.fit(X_train_resampled_no_gc, y_train_resampled, epochs=40, validation_split=0.2, verbose=0)

    model_with_gc = rebuild_model_with_new_input(best_model, new_input_dim=93)

    # Compile model_with_gc with a new optimizer to avoid variable mismatch
    model_with_gc.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    history_with_gc = model_with_gc.fit(X_train_resampled_with_gc, y_train_resampled, epochs=20, validation_split=0.2, verbose=0)

    y_pred_no_gc = best_model.predict(X_test_no_gc).flatten()
    y_pred_with_gc = model_with_gc.predict(X_test_with_gc).flatten()

    acc_no_gc = accuracy_score(y_test, np.round(y_pred_no_gc))
    auc_no_gc = roc_auc_score(y_test, y_pred_no_gc)
    precision_no_gc = precision_score(y_test, np.round(y_pred_no_gc))
    recall_no_gc = recall_score(y_test, np.round(y_pred_no_gc))
    f1_no_gc = f1_score(y_test, np.round(y_pred_no_gc))

    # Confusion matrix to calculate specificity
    tn_no_gc, fp_no_gc, fn_no_gc, tp_no_gc = confusion_matrix(y_test, np.round(y_pred_no_gc)).ravel()
    specificity_no_gc = tn_no_gc / (tn_no_gc + fp_no_gc)

    acc_with_gc = accuracy_score(y_test, np.round(y_pred_with_gc))
    auc_with_gc = roc_auc_score(y_test, y_pred_with_gc)
    precision_with_gc = precision_score(y_test, np.round(y_pred_with_gc))
    recall_with_gc = recall_score(y_test, np.round(y_pred_with_gc))
    f1_with_gc = f1_score(y_test, np.round(y_pred_with_gc))

    # Confusion matrix to calculate specificity
    tn_with_gc, fp_with_gc, fn_with_gc, tp_with_gc = confusion_matrix(y_test, np.round(y_pred_with_gc)).ravel()
    specificity_with_gc = tn_with_gc / (tn_with_gc + fp_with_gc)

    results[name] = {
        "Accuracy Without GC": acc_no_gc,
        "AUROC Without GC": auc_no_gc,
        "Precision Without GC": precision_no_gc,
        "Recall Without GC": recall_no_gc,
        "F1 Score Without GC": f1_no_gc,
        "Specificity Without GC": specificity_no_gc,
        "Accuracy With GC": acc_with_gc,
        "AUROC With GC": auc_with_gc,
        "Precision With GC": precision_with_gc,
        "Recall With GC": recall_with_gc,
        "F1 Score With GC": f1_with_gc,
        "Specificity With GC": specificity_with_gc
    }

# Convert results to DataFrame
results_df = pd.DataFrame(results).T  # Transpose the results dictionary to make cell lines rows

# Save the DataFrame to a CSV file
results_df.to_csv("performance_comparison_with_specificity.csv", index=True)

print("\nResults saved to 'performance_comparison_with_specificity.csv'")





# Sample 100 background samples for SHAP (can be adjusted)
background = X_train_resampled_with_gc[np.random.choice(X_train_resampled_with_gc.shape[0], 100, replace=False)]

# Select some test samples for explanation
test_sample = X_test_with_gc[:50]  # Use 50 test examples

# Ensure correct shape (2D: [samples, features])
assert test_sample.shape == (50, 93)




explainer = shap.DeepExplainer(model_with_gc, background)
shap_values = explainer.shap_values(test_sample)





# Use KernelExplainer or DeepExplainer depending on model type
explainer = shap.DeepExplainer(model_with_gc, X_train_resampled_with_gc[:100])  # Background data

# Explain a few test examples
shap_values = explainer.shap_values(X_test_with_gc[:50])  # 50 test samples




X_with_gc = np.hstack([X_encoded, df['GC_Content'].values.reshape(-1, 1)])





# Ensure you pass the raw model input (93 features)
background = X_train_resampled_with_gc[:100]

# Use shap.Explainer (auto-detects Deep or Kernel)
explainer = shap.Explainer(model_with_gc, background)

# Explain test instances
shap_values = explainer(X_test_with_gc[:50])

# Check feature dimensions
print("SHAP values shape:", shap_values.values.shape)  # Expect (50, 93)




# GC_Content is the last feature (index 92)
gc_shap = shap_values.values[:, 92]

print("Mean SHAP value for GC_Content:", np.mean(gc_shap))




# Use DeepExplainer for the neural network model
explainer = shap.DeepExplainer(model_with_gc, background)

# Compute SHAP values for the first test sample (for example, a single sgRNA sequence)
shap_values_single_sample = explainer.shap_values(X_test_with_gc[0:1])  # X_test_with_gc[0:1] represents one sample

# Check the shape of the SHAP values for this sample
print("SHAP values shape for a single sample:", shap_values_single_sample[0].shape)

# Since the input has 93 features, let's inspect how the GC content is represented
# Print the SHAP values for the first prediction sample to check their structure
print("SHAP values for the first test sample:", shap_values_single_sample[0])

# Now, if we are still unsure, let's print a snippet to figure out the index for GC content
print("First few SHAP values for the first sample:", shap_values_single_sample[0][0:10])  # Print the first few SHAP values

# Try to find the correct index for GC content by examining the SHAP value corresponding to it
# Assuming GC content is the last feature in your dataset
gc_shap_single_sample = shap_values_single_sample[0][0, -1]  # Index -1 assumes GC content is the last feature

# Print the contribution of GC content to the prediction for this single sgRNA sequence
print("SHAP value for GC content in the first sgRNA sequence:", gc_shap_single_sample)


# # SHAP explainer for GC content




# Example SHAP values and test sample indices (replace these with your actual values)
# shap_values_gc = array of SHAP values for GC content across all test samples
# test_sample_indices = array of test sample indices (optional)

# Replace this with your actual SHAP values array 
# Example dummy data for visualization
shap_values_gc = np.random.uniform(-0.1, 0.2, size=100)  # 100 test samples

# Create figure
plt.figure(figsize=(8, 5))
bars = plt.bar(range(len(shap_values_gc)), shap_values_gc,
               color=['red' if val > 0 else 'blue' for val in shap_values_gc])

# Plot formatting
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
# plt.title("SHAP Value Contribution of GC Content Across Test Samples")
plt.xlabel("Test Samples")
plt.ylabel("SHAP Value for GC Content")
plt.xticks(ticks=range(0, len(shap_values_gc), 5), labels=range(0, len(shap_values_gc), 5), rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', linewidth=0.5)

# Optional: Add color legend
red_patch = plt.Line2D([0], [0], color='red', lw=4, label='Positive Contribution')
blue_patch = plt.Line2D([0], [0], color='blue', lw=4, label='Negative Contribution')
plt.legend(handles=[red_patch, blue_patch])
plt.savefig("gc_content_shap_values.png", dpi=300, bbox_inches='tight')
plt.show()




plt.savefig("SHAP for gc count")

