# okcupid-gender-analysis
**Project for analyzing gender-specific language in OkCupid profiles using NLP techniques**

**Overview:**

This Project performs a series of data preprocessing and machine learning tasks on OkCupid profile text data, focusing on predicting gender based on free-text responses and then clustering users into various groups. Here's a summary of what each section accomplishes:

**Project Steps:**

* Library Loading and Setup: Loads necessary libraries for text processing, data manipulation, machine learning, and visualization.

* Data Import and Initial Setup: Imports the OkCupid dataset, selects text columns, and merges them into a single string per profile.

* Data Cleaning: Removes certain stopwords, HTML symbols, and other irrelevant strings. Lengths of the cleaned strings are stored for tracking.

* Tokenization and Further Cleaning: Tokenizes the text data, converts to lowercase, removes stopwords, performs stemming, and removes single-letter words. Each processing step reduces the data size.

* Document-Term Matrix Creation: Constructs a document-term frequency matrix (DFM), removes words with long strings (potentially meaningless), and applies TF-IDF transformation.

* Variance Thresholding: Eliminates columns with low variance, retaining only high-variance columns in the DFM.

* Data Preparation: Converts the DFM to a dense format for modeling and appends gender labels.

* Model Training: Trains a decision tree (using rpart) on the processed dataset to predict gender, saves the model, and plots the decision tree.

* Model Evaluation: Predicts on the test set and generates a confusion matrix to evaluate the model's performance.

* Removing Gender-Biased Words: Identifies words associated with male and female classifications based on decision tree splits. These words are removed from the dataset to reduce gender prediction bias.

* Clustering: (Partially completed) Prepares to perform k-means clustering on the dataset, grouping users into different clusters.

**Project Highlights:**

This pipeline offers a comprehensive approach to text data preprocessing, model training, and bias reduction, with detailed steps for each phase. The clustering step is set up but requires further code to finalize the k-means clustering and assess cluster quality.
