
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize the text
        tokens = self.tokenizer.encode(text, bos=True, eos=True, max_length=self.max_length, truncation=True, padding="max_length")
        return torch.tensor(tokens)

# Example dataset (you should replace this with your actual dataset)
texts = [
    "Once upon a time, in a faraway land...",
    "The quick brown fox jumps over the lazy dog.",
    "Chatbots are the future of customer support."
]

# Create the dataset and dataloader
dataset = TextDataset(llama_model.tokenizer, texts)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# ======================================================================================
# ======================================================================================


import torch
import torch.nn.functional as F
from torch.optim import Adam

# Set up the optimizer
optimizer = Adam(llama_model.model.parameters(), lr=5e-5)

def train_epoch(model, dataloader, optimizer):
    model.train()  # Set model to training mode
    total_loss = 0

    for batch in dataloader:
        batch = batch.cuda()  # Ensure data is on the correct device

        # Forward pass
        logits = model.model(batch, 0)  # Model forward pass
        labels = batch[:, 1:].contiguous()  # Ignore the first token (used for input)

        # Calculate the loss (cross-entropy loss)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=llama_model.tokenizer.pad_id)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Training loop
epochs = 3  # You can adjust this
for epoch in range(epochs):
    avg_loss = train_epoch(llama_model, dataloader, optimizer)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")




# Save the model after training
torch.save(llama_model.model.state_dict(), "fine_tuned_model.pth")
