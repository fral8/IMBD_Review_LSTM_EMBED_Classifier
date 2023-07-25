import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def load_data():
    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
    BUFFER_SIZE = 10000
    BATCH_SIZE = 256
    tokenizer=info.features['text'].encoder
    # Get the train and test splits
    train_data, test_data = dataset['train'], dataset['test'],

    # Shuffle the training data
    train_dataset = train_data.shuffle(BUFFER_SIZE)

    # Batch and pad the datasets to the maximum length of the sequences
    train_dataset = train_dataset.padded_batch(BATCH_SIZE)
    test_dataset = test_data.padded_batch(BATCH_SIZE)
    return train_dataset,test_dataset,tokenizer

def create_model(tokenizer):
    embedding_dim = 64
    lstm1_dim = 64
    lstm2_dim = 32
    dense_dim = 64

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm1_dim, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm2_dim)),
        tf.keras.layers.Dense(dense_dim, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

def create_model_conv(tokenizer):
    embedding_dim = 64
    filters = 128
    kernel_size = 5
    dense_dim = 64

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
        tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(dense_dim, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


train_dataset,test_dataset,tokenizer=load_data()
model=create_model(tokenizer)

NUM_EPOCHS = 10

# Train the model
history = model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset,verbose=2)
# Plot the accuracy and results
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
