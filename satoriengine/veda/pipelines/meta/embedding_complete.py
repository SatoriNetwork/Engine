'''

'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import SGDRegressor

###################################
# Embedding model & training setup
###################################

class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, embed_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
    def forward(self, x):
        return self.fc(x)

class TripletDataset(Dataset):
    def __init__(self, X, y, positive_margin=0.1, negative_margin=0.5):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        # Sort by y for easier sampling
        sorted_indices = torch.argsort(self.y)
        self.X = self.X[sorted_indices]
        self.y = self.y[sorted_indices]
        self.positive_margin = positive_margin
        self.negative_margin = negative_margin

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        anchor_x = self.X[idx]
        anchor_y = self.y[idx]

        # Positive candidate
        pos_candidates = torch.where(torch.abs(self.y - anchor_y) < self.positive_margin)[0]
        pos_candidates = pos_candidates[pos_candidates != idx]
        if len(pos_candidates) == 0:
            # fallback
            pos_candidates = torch.tensor([idx])
        pos_idx = pos_candidates[torch.randint(len(pos_candidates), ())].item()

        # Negative candidate
        neg_candidates = torch.where(torch.abs(self.y - anchor_y) > self.negative_margin)[0]
        if len(neg_candidates) == 0:
            neg_candidates = torch.arange(len(self.X))
        neg_idx = neg_candidates[torch.randint(len(neg_candidates), ())].item()

        return anchor_x, self.X[pos_idx], self.X[neg_idx]

def train_embedding_model(X, y, embed_dim=16, epochs=5, batch_size=32, lr=0.001):
    dataset = TripletDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = EmbeddingNet(X.shape[1], embed_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=1.0)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for anchor_x, pos_x, neg_x in dataloader:
            optimizer.zero_grad()
            anchor_embed = model(anchor_x)
            pos_embed = model(pos_x)
            neg_embed = model(neg_x)

            loss = criterion(anchor_embed, pos_embed, neg_embed)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

    return model

###################################
# Example usage with Iris data
###################################
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    data = load_iris()
    # Take first two features as X, and petal length (3rd feature) as score y
    X = data.data[:, :2]
    y = data.data[:, 2]

    # Train embedding model
    model = train_embedding_model(X, y, embed_dim=16, epochs=5)

    # Obtain embeddings for training data
    model.eval()
    with torch.no_grad():
        embeddings = model(torch.tensor(X, dtype=torch.float32))
        embeddings = embeddings.numpy()

    # Train an incremental regressor on embeddings
    # We'll do a simple partial_fit in a loop to simulate incremental training
    regressor = SGDRegressor(random_state=42, max_iter=1, warm_start=True)

    # Initial training: we need at least one partial_fit call with all classes if classification,
    # but here it's regression, so we can just partial_fit on a batch basis
    batch_size = 32
    for i in range(0, len(X), batch_size):
        batch_embeddings = embeddings[i:i+batch_size]
        batch_scores = y[i:i+batch_size]
        regressor.partial_fit(batch_embeddings, batch_scores)

    # Now we have a pipeline:
    # X_new -> model -> embedding_new -> regressor -> predicted_score

    X_new = np.array([[5.0, 3.5]])  # some new hypothetical observation
    with torch.no_grad():
        embedding_new = model(torch.tensor(X_new, dtype=torch.float32))
        embedding_new_np = embedding_new.numpy()
    score_pred = regressor.predict(embedding_new_np)
    print("Predicted score for X_new:", score_pred)

    # Incremental updates:
    # Suppose we get new data X_new_batch, y_new_batch and want to update both:
    X_new_batch = np.random.uniform(4, 7, size=(10, 2))  # Random new samples
    y_new_batch = np.random.uniform(1, 6, size=(10,))    # Some new scores

    # Update embeddings for the new data
    with torch.no_grad():
        new_embeddings = model(torch.tensor(X_new_batch, dtype=torch.float32)).numpy()

    # Incrementally update the regressor
    regressor.partial_fit(new_embeddings, y_new_batch)

    # Now the regressor incorporates new data as well
    # If you wanted to also update the embedding model incrementally, you could
    # call train_embedding_model again with old+new data (or just new data),
    # or implement a more sophisticated incremental training loop.
