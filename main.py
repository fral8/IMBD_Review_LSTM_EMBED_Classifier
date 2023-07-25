import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import io
def load_data():
    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    train_data, test_data = imdb['train'], imdb['test']

    # Initialize sentences and labels lists
    training_sentences = []
    training_labels = []

    testing_sentences = []
    testing_labels = []

    # Loop over all training examples and save the sentences and labels
    for s, l in train_data:
        training_sentences.append(s.numpy().decode('utf8'))
        training_labels.append(l.numpy())

    # Loop over all test examples and save the sentences and labels
    for s, l in test_data:
        testing_sentences.append(s.numpy().decode('utf8'))
        testing_labels.append(l.numpy())

    # Convert labels lists to numpy array
    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)
    return testing_sentences,testing_labels_final,training_sentences,training_labels_final

def preprocessing(sentences, padding_type, OOV_token,test_sentences,max_len,vocab_size):
    tokenizer=tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size,oov_token=OOV_token)
    tokenizer.fit_on_texts(sentences)
    sequences=tokenizer.texts_to_sequences(sentences)
    padded=tf.keras.preprocessing.sequence.pad_sequences(sequences,maxlen=max_len,truncating=padding_type)
    test_sequences=tokenizer.texts_to_sequences(test_sentences)
    test_padded=tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_len,truncating=padding_type)

    return padded,test_padded

def create_model(vocab_size,embedding_dim,maxlen):
    model=tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def save_for_embedding(model,tokenizer,vocab_size):
    # Get the embedding layer from the model (i.e. first layer)
    embedding_layer = model.layers[0]

    # Get the weights of the embedding layer
    embedding_weights = embedding_layer.get_weights()[0]
    reverse_word_index = tokenizer.index_word

    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')

    # Initialize the loop. Start counting at `1` because `0` is just for the padding
    for word_num in range(1, vocab_size):
        # Get the word associated at the current index
        word_name = reverse_word_index[word_num]

        # Get the embedding weights associated with the current index
        word_embedding = embedding_weights[word_num]

        # Write the word name
        out_m.write(word_name + "\n")

        # Write the word embedding
        out_v.write('\t'.join([str(x) for x in word_embedding]) + "\n")

    # Close the files
    out_v.close()
    out_m.close()

testing_sentences,testing_labels_final,training_sentences,training_labels=load_data()

training_sentences,testing_sentences=preprocessing(training_sentences,'post','<OOV>',testing_sentences,120,10000)

model=create_model(10000,16,120)

history=model.fit(training_sentences,training_labels,epochs=100, validation_data=(testing_sentences, testing_labels_final))