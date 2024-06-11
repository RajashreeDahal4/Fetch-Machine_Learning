### How to Run
To run the data pipeline, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/rajashreeDahal4/Fetch-THA.git
cd Fetch-THA
```

You will need the following installed on your local machine
1. docker -- docker [install guide](https://docs.docker.com/get-docker/)
2. Psql - [install](https://www.postgresql.org/download/)
3. python3

Setup python virtual environmet
```bash
python3 -m venv env
```

Activate environment for Linux/macos
```bash
source env/bin/activate
```

Activate enviornment for Windows
```
env\Scripts\activate
```

Install python dependencies
```bash
pip install -r requirements.txt
```

3. Run docker containers:
```bash
docker compose up -d
```
4. Run main python program
```bash
python3 app/main.py
```
The main.py script contains the logic to  fetch, process, and store messages.



### Task1 Sentence Transformer Implementation
Implement a sentence transformer model using any deep learning framework of your choice. 
This model should be able to encode input sentences into fixed-length embeddings.
Test your implementation with a few sample sentences and showcase the obtained embeddings. 
Describe any choices you had to make regarding the model architecture outside 
of the transformer backbone.
ANSWER: We have used pre-trained BERT model (bert-base-uncased) and its associated tokenizer. 
BERT is a widely used transformer-based model known for its effectiveness in 
natural language understanding tasks.
Regarding choices outside of transformer architecture: 
Tokenization: The choice of tokenization strategy and 
special tokens ([CLS], [SEP]) is crucial for compatibility with BERT.
Embedding Extraction: The code selects the [CLS] token embedding from the last layer. 
This is a common choice, but depending on the downstream task, different strategies 
for embedding extraction might be more suitable (e.g., averaging embeddings, using other layers, 
or pooling strategies).

### Task2 Multi-Task Learning Expansion 
Expand the sentence transformer to handle a multi-task learning setting.
Task A: Sentence Classification – Classify sentences into predefined classes (you can make these up).
Task B: [Choose another relevant NLP task such as Named Entity Recognition, Sentiment Analysis, etc.] (you can make the labels up)
Describe the changes made to the architecture to support multi-task learning.
Answer:  In our case, we considered the same dataset but used for different tasks, one is for classification, and another is for sentiment analysis.
Task2: To expand the sentence transformer for multi-task learning, several modifications are made to the architecture and training loop. 
Let's describe the changes made:
A. Model Architecture Changes:
Additional Task Layers: For each task, additional fully connected layers (fc_taskA and fc_taskB) are added to the model to handle the specific task requirements.
Forward Method Modification: The forward method now takes an additional argument task_id to specify which task's output is required. Depending on the task, the model calculates the logits and probabilities accordingly.
Training Loop Changes:
* Batch Processing: The training loop processes batches of data containing input sentences, attention masks, and labels for both tasks.
* Forward Pass: For each batch, the model computes probabilities for both tasks (probs_taskA and probs_taskB) using the specified task_id.
* Loss Computation: Separate loss functions (criterion_taskA and criterion_taskB) are applied to compute the loss for each task.
* Backpropagation and Optimization: The total loss is calculated as the sum of losses for both tasks. Backpropagation is performed, and the optimizer updates the model parameters based on the total loss.
Task-specific Output Handling:
* For Task A (Sentence Classification), a softmax activation function is applied to the logits to obtain class probabilities (probs_taskA).
* For Task B (Sentiment Analysis), a sigmoid activation function is applied to the logits to obtain binary sentiment probabilities (probs_taskB).

### Task 3: Training Considerations

Discuss the implications and advantages of each scenario and explain your rationale as to how the model should be trained given the following:
# answer:
#### Case 1: If the entire network should be frozen
This means that model's parameters remain unchanged, meaning no further learning from the new data occurs.
The output of the model is entirely dependent on the pre-trained weights. Suitable when the pre-trained model is already well-tuned for the new tasks or the new dataset is very small. The advantage is of fast inference as no backpropagation and weight updates are needed. This also prevents overfitting on small datasets since no new learning occurs. Such scenario should be rarely used in practice as the specific tasks (Classification, and sentiment analysis) likely requires some degree of adaptation to the new dataset. It is only suitable in the scenarios where computational resources are very limited, or when the pre-trained model is expected to generalize well without further tuning.
Advantages:

#### Case 2: If only the transformer backbone should be frozen
In our case, the transformer backbone (which is BERT) retains its pre-trained weights and does not adapt to the new tasks. Only the task-specific heads (the fully connected layers) are trained. This reduces the risk of overfitting as the transformer backbone retains its generalized knowledge. The training time is faster as the majority of model's parameteres are not updated. This scenario is beneficial when the pre-trained model captures general language representations well, but the task-specific heads need to learn to map these representations to the specific tasks. It is generally suitable for scenarios where the new dataset is relatively small, but task-specific tuning is still needed.

#### Case 3: If only one of the task-specific heads should be frozen
In this case, one task-specific head is retained with its pre-trained weights, whereas the other task-specific head and the transformer backbone are trained. It allows for the model to specialize in the frozen task while adapting to the other task.
It ensures that the frozen task-specific head maintains its performance while allowing flexibility for the other head to adapt. It provides a balance between retaining knowledge for one task and learning new information for the other. This scenario is useful if there is confidence that one of the task-specific heads is already well-tuned for its task, and only the other task or the backbone requires further adaptation. It is suitable when the two tasks are related, and transfer learning can help improve performance on the new task without degrading performance on the frozen task.

Transfer Learning Approach
Choice of a Pre-trained Model:

Use a model like bert-base-uncased from the Huggingface Transformers library.
This model is chosen because it has been pre-trained on a large corpus and captures a wide range of linguistic patterns.
Layers to Freeze/Unfreeze:

Freeze the transformer backbone initially: This retains the generalized language understanding captured during the pre-training phase. The backbone has been pre-trained on a large corpus and captures general language patterns, which are useful for a wide range of tasks. Freezing the backbone helps in retaining these patterns and prevents overfitting when the new dataset is relatively small.

Unfreeze and train the task-specific heads: These heads need to learn mappings from the generalized embeddings to the specific task outputs. These layers are responsible for the final task-specific predictions, so they need to adapt to the nuances of the new tasks.
By training only the task-specific heads initially, we allow the model to specialize in these tasks without disrupting the generalized language understanding of the backbone.


# Task 4: Layer-wise Learning Rate Implementation (BONUS)

Implement layer-wise learning rates for the multi-task sentence transformer.

Explain the rationale for the specific learning rates you've set for each layer.

Describe the potential benefits of using layer-wise learning rates for training deep neural networks. Does the multi-task setting play into that?
Answer: Different learning rates are assigned to specific components of the multi-task sentence transformer model:
Embeddings: Assigned a low learning rate (1e-5) to retain pre-trained knowledge.
Task-Specific Layers: Higher learning rates (1e-3) allow quicker adaptation to specific tasks.
Transformer Layer: Lower learning rate (1e-5) stabilizes training and prevents forgetting.
Benefits:

Improved Convergence: Allows different parts of the model to adapt at different rates.
Preventing Forgetting: Lower rates for pre-trained layers prevent loss of prior knowledge.
Task-Specific Adaptation: Higher rates for task-specific layers lead to quicker task adaptation.
Efficient Training: Optimizes training efficiency and resource utilization.
In multi-task learning, this approach enables flexible adaptation to task requirements while leveraging shared representations effectively.