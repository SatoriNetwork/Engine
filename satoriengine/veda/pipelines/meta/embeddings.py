import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Example embedding model: simple MLP
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

# Example dataset: we assume X is a NumPy array, y is a NumPy array
# X.shape = [num_samples, num_features], y.shape = [num_samples]
class TripletDataset(Dataset):
    def __init__(self, X, y, positive_margin=0.1, negative_margin=0.5):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.positive_margin = positive_margin
        self.negative_margin = negative_margin

        # For efficient sampling, sort by score or build an index
        sorted_indices = torch.argsort(self.y)
        self.X = self.X[sorted_indices]
        self.y = self.y[sorted_indices]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        anchor_x = self.X[idx]
        anchor_y = self.y[idx]

        # Find a positive example: similar score
        # This is a simplistic approach: binary search for neighbors with similar score
        pos_candidates = torch.where(torch.abs(self.y - anchor_y) < self.positive_margin)[0]
        pos_candidates = pos_candidates[pos_candidates != idx]
        if len(pos_candidates) == 0:
            # fallback, pick something close anyway
            pos_candidates = torch.tensor([idx])
        pos_idx = pos_candidates[torch.randint(len(pos_candidates), ())]
        pos_idx = pos_idx.item()  # Now pos_idx is an integer

        # Find a negative example: different score
        neg_candidates = torch.where(torch.abs(self.y - anchor_y) > self.negative_margin)[0]
        if len(neg_candidates) == 0:
            # fallback: pick a random point
            neg_candidates = torch.arange(len(self.X))
        neg_idx = neg_candidates[torch.randint(len(neg_candidates), ())]
        neg_idx = neg_idx.item()  # Now neg_idx is an integer

        return anchor_x, self.X[pos_idx], self.X[neg_idx]

# Example training function
def train_embedding_model(X, y, embed_dim=16, epochs=10, batch_size=32, lr=0.001):
    dataset = TripletDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = EmbeddingNet(X.shape[1], embed_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Using built-in triplet margin loss
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


def load_iris_2d_score():
    from sklearn.datasets import load_iris
    import numpy as np
    data = load_iris()
    X = data.data[:, :2]    # first two features: sepal length, sepal width
    y = data.data[:, 2]     # use petal length as the score
    return X, y


# Example usage:
if __name__ == "__main__":
    # Dummy data for illustration
    num_samples = 1000
    num_features = 20

    #rng = np.random.default_rng(37)
    #X = rng.standard_normal(size=(num_samples, num_features))
    #y = rng.standard_normal(num_samples)  # Score between 0 and 1
    X, y = load_iris_2d_score()
    print("X shape:", X.shape)  # should be (150, 2)
    print("y shape:", y.shape)  # should be (150,)
    # X now has 2D features from a real-world dataset
    # y is a continuous score (petal length) that correlates with X
    #model = train_embedding_model(X, y, embed_dim=16, epochs=20)

    # Now model maps features -> embedding.
    # For a new observation x_new:
    # embedding = model(torch.tensor(x_new, dtype=torch.float32).unsqueeze(0))

    for i in range(200):
        model = train_embedding_model(X, y, embed_dim=16, epochs=i)
        # Visualize
        import torch
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        # Suppose you have your model and the original dataset X, y
        model.eval()
        with torch.no_grad():
            embeddings = model(torch.tensor(X, dtype=torch.float32))
            embeddings = embeddings.numpy()  # convert to NumPy
        # Reduce embeddings to 2D for visualization
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        plt.figure(figsize=(8,6))
        plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], c=y, cmap='viridis')
        plt.colorbar(label='Score')
        plt.title('Embeddings Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        output_path = './visualize.png'
        plt.savefig(output_path)
        # Close the plot to free memory
        plt.close()
