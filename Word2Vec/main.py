import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import pandas as pd 
import numpy as np 

import re

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter(
#     log_dir="runs/word2vec"
# )


from pathlib import Path

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device

# vocab_size = 5000
# feature_size = 50

def encoder(path, vocab_limit) : 

    folder_path = Path(path)
    txt_files = sorted(folder_path.glob("*.txt"))   


    word_tokenised_sentences = []
    all_words = []
    encoded_sentences = []
    
    for file_path in txt_files:
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            text = re.sub(r"\s+", " ", text).strip()
            sentences = sent_tokenize(text.lower())

        stop_words = set(stopwords.words("english"))


        for sentence in sentences : 
            word_tokenised_sentence = []
            for word in word_tokenize(sentence):
                if word not in stop_words and word.isalpha():
                    word_tokenised_sentence.append(word)
                    all_words.append(word)
                    
            word_tokenised_sentences.append(word_tokenised_sentence)

        
    all_words = pd.Series(all_words)
    freq_words = all_words.value_counts().head(vocab_limit -1).index.to_list()
    freq_words = ["<UNK>"] + freq_words

    word_index_dict = {word : idx for idx,word in enumerate(freq_words)}

    for sentence in word_tokenised_sentences :
        encoded_sentence = []
        for word_token in sentence:
            encoded_sentence.append(
                word_index_dict.get(word_token,word_index_dict["<UNK>"])
            )
        encoded_sentences.append(encoded_sentence)

    return word_index_dict, encoded_sentences    

  

class Cbag_Dataset(Dataset):

    def __init__(self, encoded_sentences, window_size = 4):

        self.samples = []

        for sentence in encoded_sentences :
            for i in range(window_size, len(sentence) - window_size) : 
                target_word = torch.tensor(
                    sentence[i]
                    )

                context_words = torch.tensor(
                    sentence[ i-window_size : i ] + sentence[ i+1  :  i+window_size+1 ]
                    ) 

                self.samples.append((context_words, target_word))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        context, target = self.samples[idx]
        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(target, dtype=torch.long)
        )


def make_train_loader(encoded_sentences,window_size, batch_size):

    dataset = Cbag_Dataset(encoded_sentences,window_size)
    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
    )
    return train_dataloader

class simpleW2V(nn.Module):

    def __init__(self, vocab_count, feature_count, device = None):
        super().__init__()

        if not device :
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = nn.Embedding(
            num_embeddings = vocab_count,
            embedding_dim = feature_count,
            device= device
        )

        self.output = nn.Linear(
            in_features=feature_count,
            out_features=vocab_count,
            device= device
        )


    def forward(self, x):
        # x shape = (batch_size, context_words)
        
        features_x = self.embedding(x)   # shape = (batch_size, context_words, feature_count)
        features_x = features_x.mean(dim = 1)         # shape = (batch_size, feature_count)
        logits  = self.output(features_x)

        
        return logits 



# model = simpleW2V(vocab_size, feature_size)
# model.to(device)

# criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(
#     model.parameters(),
#     lr=0.001
# )



# print("Vocabulary:", len(word_index_dict))
# print("Sentences:", len(encoded_sentences))
# print("Training samples:", len(dataset))
# print("Batches:", len(train_dataloader))

# contexts, targets = next(iter(train_dataloader))

# print("Contexts:", contexts.shape)
# print("Targets:", targets.shape)


def training_loop(model, train_dataloader, optimizer, epochs, criterion,device = None, writer=None):
    if not device :
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = epochs
    global_step = 0

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for context_words, target_word in train_dataloader:

            context_words = context_words.to(device)
            target_word = target_word.to(device)

            logits = model(context_words)

            # print("context_words:", context_words.shape)
            # print("logits:", logits.shape)
            # print("target_word:", target_word.shape)

            loss = criterion(logits, target_word)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if writer:
                writer.add_scalar(
                    "Loss/Batch",
                    loss.item(),
                    global_step
                )
                global_step+=1


        avg_loss = total_loss / len(train_dataloader)


        if writer:
            writer.add_scalar(
                "Loss/Epoch",
                avg_loss,
                epoch
            )


        print(f"EPOCH : {epoch}, avg-loss : {avg_loss}")


def get_weight_vector(model,w_type, idx):
    if w_type == 'in' :
        w_matrix = model.embedding.weight.detach().cpu()
    else:
        w_matrix = model.output.weight.detach().cpu()

    return w_matrix[idx]


def similarity(model,w_type,word1, word2, word_index_dict) :
    word1_idx = word_index_dict[word1]
    word2_idx = word_index_dict[word2]
    w1 = get_weight_vector(model, w_type ,word1_idx)
    w2 = get_weight_vector(model, w_type ,word2_idx)

    return F.cosine_similarity(
        w1.unsqueeze(0),
        w2.unsqueeze(0)
    ).item()


def most_similar(model, word, word_index_dict, matrix_type="in", top_k=10):

    if word not in word_index_dict:
        raise ValueError(f"{word} is not in vocabulary")

    if matrix_type == "in":
        W = model.embedding.weight.detach().cpu()

    elif matrix_type == "out":
        W = model.output.weight.detach().cpu()

    else:
        raise ValueError("matrix_type must be 'in' or 'out'")

    target_idx = word_index_dict[word]
    target_vector = W[target_idx]

    similarities = F.cosine_similarity(
        W,
        target_vector.unsqueeze(0),
        dim=1
    )

    values, indices = torch.topk(
        similarities,
        k=top_k + 1
    )

    idx_to_word = {
        idx: word
        for word, idx in word_index_dict.items()
    }

    results = []

    for score, idx in zip(values, indices):

        candidate_word = idx_to_word[idx.item()]

        # skip the word itself
        if candidate_word == word:
            continue

        results.append(
            (candidate_word, score.item())
        )

        if len(results) == top_k:
            break

    return results


def save_model(name):
    torch.save({
    "model" : model.state_dict(),
    'window_size' : 2,
    'lr' : 0.001,
    'optimizer' : 'adam',
    'vocab_size' : vocab_size,
    'feature_size' : feature_size
    }, name
)




# encoded_sentences