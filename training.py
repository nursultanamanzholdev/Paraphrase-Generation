import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from torch.utils.data import DataLoader, TensorDataset
import json
from tqdm.auto import tqdm
from transformers import Adafactor


def encode_text(tokenizer, data):
    original_sentences_ids = []
    paraphrased_sentences_ids = []
    for instance in data:
        source = tokenizer(instance["Text"], return_tensors="pt", max_length=80, padding="max_length", truncation=True)
        target = tokenizer(instance["Paraphrase"], return_tensors="pt", max_length=200, padding="max_length", truncation=True)
        original_sentences_ids.append(source["input_ids"])
        paraphrased_sentences_ids.append(target["input_ids"])
    original_sentences_tensors = torch.cat(original_sentences_ids, dim=0)
    paraphrased_sentences_tensors = torch.cat(paraphrased_sentences_ids, dim=0)
    return original_sentences_tensors, paraphrased_sentences_tensors


def load_data(file_path, tokenizer):
    file = open(file_path, "r", encoding="utf-8")
    data = json.load(file)
    file.close()
    return encode_text(tokenizer, data)


def train(model, train_loader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            original_sentence_tokenized, paraphrases_tokenized = batch
            original_sentence_tokenized = original_sentence_tokenized.to("cuda")
            paraphrases_tokenized = paraphrases_tokenized.to("cuda")
            optimizer.zero_grad()
            resulted_outputs = model(input_ids=original_sentence_tokenized, labels=paraphrases_tokenized)
            loss = resulted_outputs.loss
            loss.backward()
            optimizer.step()


tokenizer = T5TokenizerFast.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base").to("cuda")

train_original_sentence, train_paraphrased_sentence = load_data("/kaggle/input/last-training/training_dataset.json", tokenizer)
train_dataset = TensorDataset(train_original_sentence, train_paraphrased_sentence)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optimizer = Adafactor(
    model.parameters(),
    lr=4e-5,
    eps=(1e-30, 0.001),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False
)

num_epochs = 5

train(model, train_loader, optimizer, num_epochs)

model.save_pretrained("/kaggle/working/final_model")
tokenizer.save_pretrained("/kaggle/working/final_model")
