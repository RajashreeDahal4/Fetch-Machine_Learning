"""
Task 2: Multi-Task Learning Expansion 

Expand the sentence transformer to handle a multi-task learning setting.

Task A: Sentence Classification – Classify sentences into predefined classes (you can make these up).
Task B: [Choose another relevant NLP task such as Named Entity Recognition, Sentiment Analysis, etc.] (you can make the labels up)
Describe the changes made to the architecture to support multi-task learning.

Task4:  Layer-wise Learning Rate Implementation (BONUS)
Implement layer-wise learning rates for the multi-task sentence transformer.
Explain the rationale for the specific learning rates you've set for each layer.
Describe the potential benefits of using layer-wise learning rates for training deep neural networks. Does the multi-task setting play into that?
Answer: 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_classes_taskA, num_classes_taskB):
            super(MultiTaskModel, self).__init__()
            # Load the pre-trained BERT model
            self.encoder = BertModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            
            # Task A: Sentence Classification
            self.fc_taskA = nn.Linear(hidden_size, num_classes_taskA)
            
            # Task B: Sentiment Analysis
            self.fc_taskB = nn.Linear(hidden_size, num_classes_taskB)

    def forward(self, input_ids, attention_mask, task_id):
        # Get the encoded representation from BERT
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoded_output = outputs.last_hidden_state[:, 0, :]  # Get the [CLS] token representation
        if task_id == 'TaskA':
            # Task A: Sentence Classification
            logits_taskA = self.fc_taskA(encoded_output)
            probs_taskA = F.softmax(logits_taskA, dim=1)
            return probs_taskA
        elif task_id == 'TaskB':
            # Task B: Sentiment Analysis
            logits_taskB = self.fc_taskB(encoded_output)
            probs_taskB = torch.sigmoid(logits_taskB)
            return probs_taskB
        else:
            raise ValueError("Invalid task_id. Choose from 'TaskA' or 'TaskB'.")

# Define your training loop (unchanged)
def train(model, dataloader, optimizer, criterion_taskA, criterion_taskB):
    model.train()
    total_loss_taskA = 0.0
    total_loss_taskB = 0.0
    
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels_taskA, labels_taskB = batch['input_ids'],batch['attention_mask'],batch['labels_taskA'],batch['labels_taskB']        
        # Forward pass
        probs_taskA= model(input_ids, attention_mask,task_id="TaskA")
        probs_taskB=model(input_ids,attention_mask,task_id="TaskB")
        
        # Compute loss for Task A
        loss_taskA = criterion_taskA(probs_taskA, labels_taskA)
        total_loss_taskA += loss_taskA.item()
        
        # Compute loss for Task B
        loss_taskB = criterion_taskB(probs_taskB, labels_taskB)
        total_loss_taskB += loss_taskB.item()
        
        # Total loss
        loss = loss_taskA + loss_taskB
        
        # Backpropagation
        loss.backward()
        optimizer.step()
    
    return total_loss_taskA / len(dataloader), total_loss_taskB / len(dataloader)

