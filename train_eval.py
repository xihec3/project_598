# import
import os
import csv
import sys
import numpy as np
import torch
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# seed
seed = 24
random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

# resource
import multiprocessing
print("CPU count: ", multiprocessing.cpu_count())
import cpuinfo
print("CPU brand: ", cpuinfo.get_cpu_info()["brand_raw"])

# hyperparameters
kBatchSize = 10  # Tried 10 and 20, 10 is better.
kEmbeddingDim = 200  # Fixed by the pre-trained word2vec 
kLearningRate = 1e-2
kNumEpochs = 15  # Determined from multiple tries
kHiddenDim = 100  # Tried 100 and 200, 100 is better.
kValidationRate = 0.05  # For choosing hyperparameters
kDataPath = "csvs/"
kBioEmbeddingPath = "biowordvec.bin"
kEmbeddingType = "med"  # med, general, random
kDataType = 0  # 0 for normal, 1 for neg2
kClassNum = 6  # for normal, this is 6; for neg2, this is 2
kF1Mode = "micro"  # micro, absent 
kTrainFileName = "bidph.txt"  # bidph.txt or bidph_neg2.txt
kTestFileName = "upmc.txt"  # upmc.txt or upmc_neg2.txt

# Load embedding
from gensim.models import KeyedVectors, Word2Vec
import gensim.downloader
word2vec = None
if kEmbeddingType == "med":
  word2vec = KeyedVectors.load_word2vec_format("biowordvec.bin", binary=True)
elif kEmbeddingType == "general":
  word2vec = gensim.downloader.load('glove-wiki-gigaword-200')
# Keep a dictionary of words and words' embeddings
word_to_idx = {}
word_embeddings = []
# Treat <c> and </c> specially
word_to_idx["<c>"] = 0
word_embeddings.append(np.random.rand(kEmbeddingDim))
word_to_idx["</c>"] = 1
word_embeddings.append(np.random.rand(kEmbeddingDim))
# Load the training set vocabulary
with open(kDataPath + kTrainFileName, "r") as f:
  reader = csv.reader(f, delimiter="\t")
  next(reader) # Skip header
  for entry in reader:
    sentence = entry[0]
    for word in sentence.split(" "):
      if word in word_to_idx:
        continue
      if kEmbeddingType == "med" or kEmbeddingType == "general":
          if word in word2vec:
            word_to_idx[word] = len(word_to_idx)
            word_embeddings.append(word2vec[word])
      else:
        word_to_idx[word] = len(word_to_idx)
        word_embeddings.append(np.random.rand(kEmbeddingDim))
# Load the testing set vocabulary
with open(kDataPath + kTestFileName, "r") as f:
  reader = csv.reader(f, delimiter="\t")
  next(reader) # Skip header
  for entry in reader:
    sentence = entry[0]
    for word in sentence.split(" "):
      if word in word_to_idx:
        continue
      if kEmbeddingType == "med" or kEmbeddingType == "general":
          if word in word2vec:
            word_to_idx[word] = len(word_to_idx)
            word_embeddings.append(word2vec[word])
      else:
        word_to_idx[word] = len(word_to_idx)
        word_embeddings.append(np.random.rand(kEmbeddingDim))
# Convert word_embeddings to tensor
word_embeddings = np.concatenate(word_embeddings).reshape(-1, kEmbeddingDim)
word_embeddings = torch.tensor(word_embeddings, dtype=torch.float)
print("Word embedding shape: ", word_embeddings.shape)
# Profile memory usage
import os, psutil
process = psutil.Process(os.getpid())
print("Memory after loading embedding: ", process.memory_info().rss/1024/1024/1024)  # in GB 
# We will not use word2vec, so gc it
del word2vec
import gc
gc.collect()

# Create data loader
# Convert a sentence to a (L, ) numpy array, L is the word in the sentence.
# The value is the idx of each word.
# Note that this will skip the word with unknown embedding. 
def ProcessSentence(sentence):
  indices = []
  for word in sentence.split(" "):
    if word in word_to_idx:
      indices.append(word_to_idx[word])
  return np.array(indices)

from torch.utils.data import Dataset

class I2b2Dataset(Dataset):
    # `embedding` can be med, random, or general.
    def __init__(self, filename):
      self.sentences = []  # Will store a list of numpy array, each representing a sentence
      self.labels = []
      with open(kDataPath + filename, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader) # Skip header
        for entry in reader:
          self.sentences.append(ProcessSentence(entry[0]))
          self.labels.append(int(entry[1]))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]

# Batch multiple sentences, create tensors, and perform masking.
# Input:
#     data contains sentences_raw and labels_raw.
#     sentences_raw is a tuple of np arrays of shape (L, ). L is the longest sentence length.
#     labels_raw is a tuple of int.
# Return:
#     sentences: Tensor of shape (N, L). Each value represents the index of a word.
#     labels: Tensor of shape (N, )
#     masks: Tensor or shape (N, L). Value=1 means that sentence has word at that position. Value=0 means that sentence does not have word at that position.
def collate_fn(data):
  sentences_raw, labels_raw = zip(*data)
  labels = torch.tensor(labels_raw, dtype=torch.long)
  N = len(sentences_raw)
  L = 0
  for s in sentences_raw:
    if s.shape[0] > L:
      L = s.shape[0]
  sentences = torch.zeros((N, L), dtype=torch.int)
  masks = torch.zeros((N, L), dtype=torch.float)
  for index_sent, sent in enumerate(sentences_raw):
        for index_word, word in enumerate(sent):
          sentences[index_sent][index_word] = torch.tensor(word, dtype=torch.float)
          masks[index_sent][index_word] = 1
  return sentences, labels, masks

# Define Att-BiLSTM
"""
MIT License

Copyright (c) 2020 Haitao Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class Att_BLSTM(nn.Module):
    def __init__(self, word_embedding, class_num, embedding_dim, hidden_dim):
        super().__init__()
        self.word_embedding = word_embedding
        self.class_num = class_num
        # Provided hyper paramter
        self.word_dim = embedding_dim
        self.hidden_size = hidden_dim
        # Fixed hyper parameters
        self.layers_num = 1
        self.emb_dropout_value = 0.3
        self.lstm_dropout_value = 0.3
        self.linear_dropout_value = 0.5

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_embedding,
            freeze=False,
        )
        self.lstm = nn.LSTM(
            input_size=self.word_dim,
            hidden_size=self.hidden_size,
            num_layers=self.layers_num,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True,
        )
        self.tanh = nn.Tanh()
        self.emb_dropout = nn.Dropout(self.emb_dropout_value)
        self.lstm_dropout = nn.Dropout(self.lstm_dropout_value)
        self.linear_dropout = nn.Dropout(self.linear_dropout_value)

        self.att_weight = nn.Parameter(torch.randn(1, self.hidden_size, 1))
        self.dense = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.class_num,
            bias=True
        )

        # initialize weight
        init.xavier_normal_(self.dense.weight)
        init.constant_(self.dense.bias, 0.)

    def lstm_layer(self, x, mask):
        lengths = torch.sum(mask.gt(0), dim=-1)
        max_length = torch.max(lengths)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, (_, _) = self.lstm(x)
        h, _ = pad_packed_sequence(h, batch_first=True, padding_value=0.0)  # (N, L, H)
        h = h.view(-1, max_length, 2, self.hidden_size)  # (N, L, 2, H)
        h = torch.sum(h, dim=2)  # (N, L, H)
        return h

    def attention_layer(self, h, mask):
        att_weight = self.att_weight.expand(mask.shape[0], -1, -1)  # Same value expanded to (N, H, 1)
        att_score = torch.bmm(self.tanh(h), att_weight)  # (N, L, H) * (N, H, 1) -> (N, L, 1)

        # mask, remove the effect of 'PAD'
        mask = mask.unsqueeze(dim=-1)  # (N, L, 1)
        att_score = att_score.masked_fill(mask.eq(0), float('-inf'))  # (N, L, 1)
        att_weight = F.softmax(att_score, dim=1)  # (N, L, 1)

        reps = torch.bmm(h.transpose(1, 2), att_weight).squeeze(dim=-1)  # (N, H, L) * (N, L, 1) -> (N, H, 1) -> (N, H)
        reps = self.tanh(reps)  # (N, H)
        return reps

    # sentences: (N, L)
    # masks: (N, L)
    def forward(self, sentences, masks):
        emb = self.word_embedding(sentences)  # (N, L, E)
        emb = self.emb_dropout(emb)
        h = self.lstm_layer(emb, masks)  # (N, L, H)
        h = self.lstm_dropout(h)
        reps = self.attention_layer(h, masks)  # (N, H)
        reps = self.linear_dropout(reps)
        logits = self.dense(reps)
        scores = F.softmax(logits, dim=1)
        return scores

# Prepare for train
model = Att_BLSTM(word_embeddings, kClassNum, kEmbeddingDim, kHiddenDim)
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=kLearningRate)

# Define evaluate
from sklearn.metrics import f1_score
def Evaluate(model, test_loader, f1_mode):
  model.eval()
  y_pred = []
  y_true = []

  for x, y, masks in test_loader:
    y_hat = model(x, masks)
    y_result = torch.argmax(y_hat, dim=1)
    for i in range(len(y)):
      y_true.append(y[i])
      y_pred.append(y_result[i])

  f = None
  if f1_mode == "micro":
    f = f1_score(y_pred=y_pred, y_true=y_true, average='micro')
  elif f1_mode == "absent":
    fs = f1_score(y_pred=y_pred, y_true=y_true, average=None)
    f = fs[1]  # The absent class
  else:
    raise Exception("Unsupported F1 mode")
  return f

# Train
import time
def Train(model, train_loader, val_loader, n_epochs, f1_mode):
  starting_time = time.time()
  for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    for x, y, masks in train_loader:
      optimizer.zero_grad()
      y_hat = model(x, masks)
      loss = criterion(y_hat, y)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
    train_loss = train_loss / len(train_loader)
    validation_f1 = Evaluate(model, val_loader, f1_mode)
    print('Epoch: {}, time elapsed: {:.1f}min \t Training Loss: {:.4f}, validation f1: {:.4f}'.format(epoch+1, (time.time()-starting_time)/60, train_loss, validation_f1))
# Split dataset into train and validation
dataset = I2b2Dataset(kTrainFileName)
val_set_size = int(len(dataset) * kValidationRate)
train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-val_set_size, val_set_size])
# number of epochs to train the model
train_loader = DataLoader(train_set, batch_size=kBatchSize, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(val_set, batch_size=kBatchSize, collate_fn=collate_fn, shuffle=False)
Train(model, train_loader, val_loader, kNumEpochs, kF1Mode)

# Evaluate
test_dataset = I2b2Dataset(kTestFileName)
test_loader = DataLoader(test_dataset, batch_size=kBatchSize, collate_fn=collate_fn, shuffle=False)
print("Train f1: {:4f}".format(Evaluate(model, train_loader, kF1Mode)))
print("Test f1: {:4f}".format(Evaluate(model, test_loader, kF1Mode)))

# Profile memory usage at last
import os, psutil
process = psutil.Process(os.getpid())
print("After train memory usage: ", process.memory_info().rss/1024/1024/1024)  # in GB 
