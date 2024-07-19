#!/usr/bin/env python
# coding: utf-8

# # setup

# In[2]:


get_ipython().system('pip install datasets tqdm')


# # data

# In[5]:


from datasets import load_dataset
dataset = load_dataset('EleutherAI/pile', split='train', streaming=True)


# In[15]:


from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function)


# In[16]:


import torch
from torch.utils.data import DataLoader, IterableDataset

class StreamingDataset(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for item in self.dataset:
            yield item

streaming_dataset = StreamingDataset(tokenized_datasets)
train_dataloader = DataLoader(streaming_dataset, batch_size=8)


# # model

# In[17]:


import torch
from torch import nn

class SimpleTransformer(nn.Module):
    def __init__(self, d_model, d_ff, vocab_size):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=d_ff, activation='relu')
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        x = self.fc(x)
        return x

# Model parameters
d_model = 128
d_ff = 512
vocab_size = 50257  # Size of tokenizer vocabulary (e.g., GPT-2 tokenizer size)

model = SimpleTransformer(d_model=d_model, d_ff=d_ff, vocab_size=vocab_size).cuda()


# In[13]:


get_ipython().system('pip install zstandard')


# In[18]:


from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

# Training parameters
num_epochs = 1
learning_rate = 5e-5
max_steps = 10000  # Set a fixed number of training steps

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=max_steps)

# Training loop
model.train()
step = 0
for epoch in range(num_epochs):
    for batch in tqdm(train_dataloader):
        if step >= max_steps:
            break

        input_ids = batch['input_ids'].cuda()
        labels = batch['input_ids'].cuda()  # Use the input as labels for language modeling

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        step += 1

    if step >= max_steps:
        break

    print(f"Epoch {epoch+1}/{num_epochs} completed. Loss: {loss.item()}")


# In[ ]:




