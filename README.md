# IMDB Review Classification

This repository contains code for text classification of IMDB movie reviews. The goal is to predict whether a movie review is positive or negative. The dataset consists of movie reviews from the Internet Movie Database (IMDB).

## Dataset

The IMDB movie reviews dataset is used for training and evaluation. The dataset contains movie reviews as text, and each review is labeled as either positive or negative sentiment.

## Main_LSTM.py

### Data Loading and Preprocessing

- The dataset is loaded using the `tensorflow_datasets` library, which provides the IMDB_reviews/subwords8k dataset with subword tokenization.
- The training and test data are shuffled and padded to a fixed length using `padded_batch`.
- The tokenizer is obtained from the dataset's information and used to convert text data to sequences and pad them.

### LSTM-based Model

- The LSTM-based model architecture consists of an Embedding layer followed by two Bidirectional LSTM layers to capture sequential information from the text.
- The model has a Dense layer for feature extraction and a final Dense layer with a sigmoid activation function for binary classification (positive or negative).

### Training and Evaluation

- The model is trained using binary cross-entropy loss and the Adam optimizer for 10 epochs.
- The training and validation accuracy and loss are plotted to assess the model's performance.

## Main.py

### Data Loading and Preprocessing

- The dataset is loaded using `tensorflow_datasets`, and the movie reviews and their corresponding labels are extracted.
- The sentences are converted to sequences using `Tokenizer`, and the sequences are padded to a fixed length.

### Embedding-based Model

- The embedding-based model architecture consists of an Embedding layer followed by a Flatten layer to flatten the sequence data.
- The model then has a Dense layer for feature extraction and a final Dense layer with a sigmoid activation function for binary classification.

### Training and Evaluation

- The model is trained using binary cross-entropy loss and the Adam optimizer for 100 epochs.
- The training and validation accuracy and loss are plotted to assess the model's performance.

## Save Embedding

- The `save_for_embedding` function is included to save the word embeddings for visualization using the Embedding Projector.

Feel free to experiment with different hyperparameters, model architectures, or other methods to further improve the classification accuracy.

For any questions or suggestions, please contact [Francesco Alotto](mailto:franalotto94@gmail.com). Happy movie review classification with AI! 🎥🤖
