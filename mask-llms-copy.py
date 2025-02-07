import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Function to mask tokens
def mask_tokens(inputs, mask_prob=0.15):
    labels = inputs.clone()
    
    # Probability of masking a token
    prob = torch.rand(inputs.shape)
    mask = (prob < mask_prob) & (inputs != tokenizer.pad_token_id) & \
           (inputs != tokenizer.cls_token_id) & (inputs != tokenizer.sep_token_id)

    inputs[mask] = tokenizer.mask_token_id
    return inputs, labels

# Define a small sample corpus
unlabeled_corpus = [
    "Machine learning is transforming the world.",
    "BERT is a powerful transformer model.",
    "Masked language modeling helps BERT understand context."
]

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(3):  # Example: 3 epochs
    for text in unlabeled_corpus:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        masked_inputs, labels = mask_tokens(inputs['input_ids'])

        outputs = model(masked_inputs, labels=labels)
        loss = outputs.loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")

print("Training complete.")
