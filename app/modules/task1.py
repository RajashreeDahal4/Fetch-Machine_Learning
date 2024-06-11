"""
Date: June 11, 2024
Name: Rajashree Dahal
Task 1: Sentence Transformer Implementation
Implement a sentence transformer model using any deep learning framework of your choice. 
This model should be able to encode input sentences into fixed-length embeddings.
Test your implementation with a few sample sentences and showcase the obtained embeddings. 
Describe any choices you had to make regarding the model architecture outside 
of the transformer backbone.
"""

from transformers import BertModel, BertTokenizer
import torch

# Load pre-trained model and tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)

sentences = [
    "This is a sample sentence 1.",
    "This is a sample sentence2.",
    "This is a sample sentence 3.",
]
# Tokenize and process each sentence
embeddings = []
for sentence in sentences:
    # Tokenize sentence
    tokens = tokenizer.tokenize(sentence)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = torch.tensor(token_ids).unsqueeze(0)

    # Pass through model
    with torch.no_grad():
        output = model(token_ids)
        embedding = output[1]  # Extract embeddings from the last layer
    embeddings.append(embedding)

# Print the embeddings shape
for i, embedding in enumerate(embeddings):
    print("Embeddings for sentence", i + 1, ":", embedding.shape)
