# RNAS-sgRNA
RNAS-sgRNA is a deep learning framework for analyzing and predicting sgRNA activity
# 📈 Pipeline Overview
# Neural Architecture Search (NAS):
Implemented using AutoKeras on the benchmark dataset.

+ Transfer Learning:
The pretrained model is fine-tuned on 4 different cell line datasets: HCT116, HeLa, HEK293, and HL60.

+ Data Balancing:
We use under-sampling to balance binary labels and validate dataset similarity using:

Kolmogorov–Smirnov (KS) test

Mann–Whitney U (MWU) test

+ Generalization to Independent Datasets:
Evaluate model performance on independent datasets such as A549.

 # 1. Neural Architecture Search (NAS) on Benchmark Dataset
File: benchmark_model.py

- AutoKeras to perform NAS on a benchmark sgRNA dataset.

- The dataset is preprocessed (e.g., sequences are one-hot encoded).

- The model is trained and saved as best_autokeras_model.h5.
# 2. Transfer Learning on Four Cell Line Datasets
File: Transfer_learning_cell_line_.py

- Loads the pretrained model (best_autokeras_model.h5).

- Applies transfer learning to four specific cell line datasets (e.g., HCT116, HELA, HEK293 and HL60.).

- SMOTE is used for data balancing to address class imbalance.

- Model is fine-tuned for each dataset using their respective training sets.

- Outputs performance metrics like ROC AUC, accuracy, etc.
# 3. Data Balancing and Statistical Validation
File: Data_balancing_ks_mwu_test.py

- Performs binary class balancing (e.g., under-sampling) to address class imbalance in efficacy labels.

- Applies Kolmogorov–Smirnov (KS) test and Mann–Whitney U (MWU) test to compare feature distributions:

- Between different datasets (e.g., benchmark vs. cell-line)

- Between efficacy classes (high vs. low)

- Visualizes the distributions using seaborn histograms and statistical tables.

- Helps justify the biological and statistical feasibility of transfer learning between datasets.
 # 4. Generalization on Independent Datasets
File: Independent_datasets_generalization_cleaned.py

- The final model (after transfer learning) is evaluated on independent test datasets like A549, K562, or NB4.

- The log2fc values are normalized and predicted.

- Pearson correlation is calculated to measure the agreement between predicted and actual efficacy values.

- The script handles potential layer mismatch (e.g., CastToFloat32) by registering custom layers.
  
# ▶️ Usage
## Train NAS model on benchmark dataset:
```sh
python benchmark_model.py
## Fine-tune with transfer learning:
```sh
python Transfer_learning_cell_line.py
## Balance datasets and run statistical tests:
```sh
python Data_balancing_ks_mwu_test.py
Generalize model on independent datasets:
```sh
python Independent_datasets_generalization.py




Markdown
@misc{crisprgraphnet2025,
  title        = {RNAS-sgRNA: Recurrent Neural Architecture Search for Detection of On-target Effects in Single Guided RNA},
  author       = {Shehla Rafiq, Assif Assad},
  year         = {2025},
  howpublished = {\url{https://github.com/shehlarafiq5/RNAS-sgRNA}},
  note         = {Unpublished manuscript}
}











