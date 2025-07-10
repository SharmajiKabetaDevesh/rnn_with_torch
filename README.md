
# Question Answering using RNN in PyTorch

This repository demonstrates a simple implementation of a **Recurrent Neural Network (RNN)** using **PyTorch** to perform question-answer learning. The model is trained on a dataset of paired questions and answers to predict a response word given a question input.

This project covers:

* Text preprocessing and tokenization
* Vocabulary creation
* Dataset preparation with PyTorch's `Dataset` and `DataLoader`
* Building and training a custom RNN model
* Making predictions using the trained model

---

## Problem Overview

Given a dataset containing question-answer pairs, the goal is to train an RNN model that can predict the correct answer word based on a question. This is a basic form of **sequence-to-one** learning, often used in conversational AI, chatbot design, or question-answering systems.

---

## What This Project Does

* Preprocesses natural language using NLTK
* Converts text to numerical format using a custom vocabulary
* Trains a PyTorch RNN model to map input questions to output answers
* Provides a simple prediction function to test the model on unseen questions

---

## Model Architecture: Simple RNN

### 1. Embedding Layer

```python
self.embedding = nn.Embedding(vocab_size, embedding_dim=50)
```

* Maps integer-encoded words into dense vector representations (embeddings).
* Helps the model learn semantic relationships between words.

### 2. RNN Layer

```python
self.rnn = nn.RNN(input_size=50, hidden_size=64, batch_first=True)
```

* Processes the embedded question sequence.
* Returns a hidden state which captures temporal (sequential) patterns in the input.

### 3. Fully Connected Output Layer

```python
self.fc = nn.Linear(64, vocab_size)
```

* Projects the final RNN hidden state into a probability distribution across the vocabulary space.
* Uses `CrossEntropyLoss` to compute error between predicted and actual answer words.

---

## Dataset Format

The dataset is a CSV file `data_rnn.csv` with two columns:

```
| question         | answer     |
|------------------|------------|
| What is AI?      | intelligence |
| Capital of India | delhi        |
```

This file is processed and turned into numeric tensors using custom preprocessing logic.

---

## Preprocessing Steps

1. Lowercasing
2. Removing punctuation
3. Tokenization using `nltk.word_tokenize`
4. Vocabulary creation across both questions and answers
5. Unknown token (`<UNK>`) added to handle out-of-vocabulary words

---

## Training Pipeline

1. Texts are converted to sequences of indices using the created vocabulary.
2. PyTorch `Dataset` and `DataLoader` are used to batch and shuffle data.
3. The model is trained using the **Adam optimizer** and **cross-entropy loss**.
4. The training loop runs for a fixed number of epochs.

Example loss output:

```
Epoch: 1, Loss: 10.345891
Epoch: 2, Loss: 9.204719
...
Epoch: 50, Loss: 0.876231
```

---

## Prediction Function

```python
predict(model, "capital of india")
```

The function:

* Converts a given question into a tensor
* Feeds it into the trained model
* Uses softmax to extract the highest probability word from the vocabulary
* Returns the predicted word or "I don't know" if confidence is low

---
![xcwe](https://github.com/user-attachments/assets/ca09d127-7392-41dc-8166-c5d4f76afc4b)
## Why Use an RNN?

RNNs are designed to process sequences. For this project, where a question is treated as a sequence of words and the answer is a single word, an RNN is a natural fit due to its ability to maintain context through time steps.

### Key Features of RNN:

* Maintains a hidden state to capture dependencies between words.
* Learns to extract meaning from the order of words.
* Works well for simple text-based question-answer tasks.

---

## Limitations and Future Improvements

* Current model only predicts **one-word answers**.
* Does not use advanced architectures like LSTM or GRU, which handle long-term dependencies better.
* No padding or batching of sequences; limited to batch size of 1.
* Model can be extended to support:

  * **Multi-word answers**
  * **Attention mechanism**
  * **Sequence-to-sequence architecture with encoder-decoder RNNs**

---

## Requirements

```bash
torch
torchvision
pandas
numpy
nltk
```

Install using:

```bash
pip install torch torchvision pandas numpy nltk
```

Make sure to download the NLTK resources before training:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## How to Use

1. Clone the repo:

```bash
git clone https://github.com/yourusername/simple-rnn-qa-pytorch.git
cd simple-rnn-qa-pytorch
```

2. Place your dataset as `data_rnn.csv`.

3. Run the notebook:

```bash
jupyter notebook RNNwithtorch.ipynb
```

4. Train the model and use `predict()` to test new questions.

---

## Conclusion

This project is a minimal working example of how a simple RNN model can be trained from scratch using PyTorch for
question-answer prediction tasks. It lays a foundation for building more advanced NLP systems such as chatbots, conversational agents, or intelligent Q\&A systems.


