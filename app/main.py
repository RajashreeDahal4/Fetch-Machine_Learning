from modules.tasks2_4 import MultiTaskModel,train 
from transformers import BertModel, BertTokenizer
from modules.dataset import MultiTaskDataset
from torch.utils.data import DataLoader 
import torch.nn as nn
import torch.optim as optim


class Trainer:
    '''
    Main class to train the multi task model with multi layered learning rate
    '''
    def __init__(self):
        self.model_name='bert-base-uncased'


    def train(self):
        model = MultiTaskModel(self.model_name,4, 2)
        # Using layers specific learning rates
        optimizer = optim.Adam(
            [
                {"params": model.encoder.embeddings.parameters(), "lr": 1e-5},
                {"params": model.fc_taskB.parameters(),"lr":1e-3},
                {"params": model.fc_taskA.parameters(), "lr": 1e-3},
                {"params": model.encoder.encoder.layer[11].parameters(), "lr": 1e-5}
            ],
            lr=5e-4,
        )
        criterion_taskA = nn.CrossEntropyLoss()
        criterion_taskB = nn.BCELoss()


        # Create synthetic data
        sentences = ["This is a positive sentence.", "This is a negative sentence."] * 100
        labels_taskA = [0, 1] * 100  # Binary classification for simplicity
        labels_taskB = [[1, 0], [0, 1]] * 100  # Sentiment analysis: [positive, negative]

        # Initialize tokenizer
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        max_len = 50

        # Create the dataset and dataloader
        dataset = MultiTaskDataset(sentences, labels_taskA, labels_taskB, tokenizer, max_len)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        # # Initialize the model, optimizer, and loss functions
        model = MultiTaskModel(self.model_name, num_classes_taskA=2, num_classes_taskB=2)
        optimizer = optim.Adam(model.parameters(), lr=2e-5)
        criterion_taskA = nn.CrossEntropyLoss()
        criterion_taskB = nn.BCELoss()

        # Run the training loop
        num_epochs = 3
        for epoch in range(num_epochs):
            loss_taskA, loss_taskB = train(model, dataloader, optimizer, criterion_taskA, criterion_taskB)
            print(f'Epoch {epoch + 1}, Loss Task A: {loss_taskA:.4f}, Loss Task B: {loss_taskB:.4f}')

    def run(self):
        self.train()
        print("Training complete, Thank you!!")

if __name__ == "__main__":
    app = Trainer()
    app.run()

