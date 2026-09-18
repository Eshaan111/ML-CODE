import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from nltk import word_tokenize, sent_tokenize
import sys
import os
from pathlib import Path
import tiktoken
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import json 
import mlflow
from torch.utils.tensorboard import SummaryWriter
project_root = Path.cwd().resolve()
print(project_root)
sys.path.insert(0, str(project_root))
from Components.MultiBlockDecoder import CompleteDecoderBlock
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




# ----------------------------------------------TOKENISERs------------------------------

tokenizer = Tokenizer.from_file(r"MODEL-Implementation\GPT-tiny-shakespere\tokenizer-8k.json")
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()
with open(r'MODEL-Implementation\GPT-tiny-shakespere\tiny shakespeare.txt', 'r') as file :
    text = file.read()
tokenised_corpus = tokenizer.encode(text)
PAD_IDX = tokenizer.token_to_id("[PAD]")


with open(r"MODEL-Implementation\GPT-tiny-shakespere\tokenizer-8k.json", "r", encoding="utf-8") as file:
    tokenizer_data = json.load(file)
vocab_dict = tokenizer_data.get("model",{}).get("vocab",{})

tokens = tokenised_corpus.ids
n = len(tokens)
train_end = int(0.90 * n)
val_end   = int(0.95 * n)

train_tokens = tokens[:train_end]
val_tokens   = tokens[train_end:val_end]
test_tokens  = tokens[val_end:]

# trainer = BpeTrainer(
#     vocab_size=8000, 
#     special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
# )

    
# files = [r'../tiny shakespeare.txt']
# tokenizer.train(files, trainer)
# tokenizer.save("tokenizer-8k.json")
# tokenizer.token_to_id('[PAD]')

config = {
    "embed_dim": 512,
    "num_heads": 8,
    "num_blocks": 6,
    "batch_size": 32,
    "lr": 3e-4,
    "epochs": 200,
    "chunk_size": 128,
    "vocab_count" : len(vocab_dict),
    "ffo_neurons" : 1024,
    "is_cross_attention": False
}


# ----------------------------------------------DATASETS------------------------------

class DatasetShakespeare(Dataset):
    def __init__(self, chunk_length, token_corpus):
        self.token_corpus = torch.tensor(token_corpus, dtype=torch.long)
        self.chunk_length = chunk_length
    def __len__(self):
        return (len(self.token_corpus)-1)//self.chunk_length
    
    def __getitem__(self, index):
        start = index * self.chunk_length
        end = start + self.chunk_length + 1
        
        window = self.token_corpus[start : end ]
        input = window[:-1]
        target = window[1:]
        
        return input, target
    
train_dataset = DatasetShakespeare(
    config["chunk_size"],
    train_tokens
)

val_dataset = DatasetShakespeare(
    config["chunk_size"],
    val_tokens
)

test_dataset = DatasetShakespeare(
    config["chunk_size"],
    test_tokens
)

# ----------------------------------------------LOADERS------------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    pin_memory=True
)

# ------------------------------------------------------MODEL-----------------------------


model = CompleteDecoderBlock(config["num_blocks"], config["num_heads"], config["vocab_count"], config["embed_dim"], config["ffo_neurons"], config["is_cross_attention"])
model.to(device)
Criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.AdamW(model.parameters(), lr = config["lr"])

total_trained_epochs = 0
global_step = 0


def token_accuracy_counts(logits, targets, pad_idx):
    predictions = logits.argmax(dim=-1)

    mask = targets != pad_idx

    correct = ((predictions == targets) & mask).sum().item()
    total = mask.sum().item()

    return correct, total


def train_one_epoch(model, optimizer, Criterion, data_loader):
    global global_step
    global total_trained_epochs
    model.train()
    epoch_loss = 0
    epoch_correct_preds = 0
    epoch_total_preds = 0
    
    for input,target in data_loader:
            input = input.to(device, non_blocking = True)
            target = target.to(device, non_blocking = True)
            optimizer.zero_grad(set_to_none = True)
            
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(input) #logits = (B,T,V)
                logits_flat = logits.flatten(0,1)
                target_flat = target.flatten()
                loss = Criterion(logits_flat, target_flat)
                correct, total = token_accuracy_counts(logits, target, PAD_IDX )
                epoch_correct_preds += correct
                epoch_total_preds += total
                
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if global_step % 30 == 0:

                writer.add_scalar(
                    "train/batch_loss",
                    loss.item(),
                    global_step
                )

                writer.add_scalar(
                    "train/learning_rate",
                    optimizer.param_groups[0]["lr"],
                    global_step
                )
            global_step+=1
            
            
    avg_loss = epoch_loss/len(data_loader)
    print(f"EPOCH : {total_trained_epochs + 1} , Avg Loss = {avg_loss}")
    total_trained_epochs +=1
    
    return {
        "loss": avg_loss,
        "token_accuracy": epoch_correct_preds / epoch_total_preds
    }




def validate_one_epoch(model, Criterion, data_loader):
    model.eval()

    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0

    with torch.no_grad():
        for input, target in data_loader:
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda"
            ):
                logits = model(input)

                loss = Criterion(
                    logits.flatten(0, 1),
                    target.flatten()
                )

            correct, total = token_accuracy_counts(
                logits,
                target,
                PAD_IDX
            )

            epoch_loss += loss.item()
            epoch_correct += correct
            epoch_total += total

    return {
        "loss": epoch_loss / len(data_loader),
        "token_accuracy": epoch_correct / epoch_total
    }
    
# LOGGING--------------------------
writer = SummaryWriter(
    log_dir="runs/decoder_E512_H8_L6"
)

# MLflow
mlflow.set_experiment("Decoder Transformer")

with mlflow.start_run(
    run_name="E512-H8-L6-lr3e-4"
):

    mlflow.log_params(config)
    for i in range(config["epochs"]):
        train_obj = train_one_epoch(model, optimizer, Criterion, train_loader)
        train_loss = train_obj["loss"] 
        train_token_accuracy = train_obj["token_accuracy"]
        writer.add_scalar("train/epoch_loss", train_loss, i)
        writer.add_scalar("train/token_accuracy", train_token_accuracy, i)
    
        val_obj = validate_one_epoch(model, Criterion, val_loader)
        val_loss = val_obj["loss"] 
        val_token_accuracy = val_obj["token_accuracy"]
        writer.add_scalar("val/epoch_loss", val_loss, i)
        writer.add_scalar("val/token_accuracy", val_token_accuracy, i)
    
    
        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_token_accuracy": train_token_accuracy,
            "val_loss": val_loss,
            "val_token_accuracy": val_token_accuracy
        }, step=i)
    
    
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config" : config,
    "train_loss": train_loss,
    "val_loss": val_loss,
    "total_trained_epochs": total_trained_epochs,
    "global_step": global_step,
    "tokenizer_file": "tokenizer-8k.json",
    
}    
    
torch.save(
    checkpoint,
    "checkpoint-module.pt"
)
writer.close()
    
    