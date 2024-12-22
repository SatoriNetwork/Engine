import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

# Assuming you have:
# model: trained embedding model
# X, y: your training data and scores

# Compute embeddings for all training points
model.eval()
with torch.no_grad():
    train_embeddings = model(torch.tensor(X, dtype=torch.float32)).numpy()

# Fit a NearestNeighbors model on the embeddings
k = 5  # choose your k
nn_model = NearestNeighbors(n_neighbors=k)
nn_model.fit(train_embeddings)

def predict_score_knn(x_new):
    """
    Given a new feature vector x_new, compute its embedding and use kNN in embedding space
    to predict the score.
    """
    with torch.no_grad():
        emb_new = model(torch.tensor(x_new, dtype=torch.float32).unsqueeze(0)).numpy()

    # Find k nearest neighbors in embedding space
    distances, indices = nn_model.kneighbors(emb_new)  # indices of neighbors
    # Average the scores of the nearest neighbors
    neighbor_scores = y[indices[0]]
    return np.mean(neighbor_scores)

# Example usage:
x_new = X[0]  # Just taking an existing point as an example
pred_score = predict_score_knn(x_new)
print("Predicted score using kNN in embedding space:", pred_score)

# Incremental Updates:
# If you get new data points (X_new_batch, y_new_batch):
with torch.no_grad():
    new_embeddings = model(torch.tensor(X_new_batch, dtype=torch.float32)).numpy()

# Append these new embeddings and scores to your arrays
train_embeddings = np.vstack([train_embeddings, new_embeddings])
y = np.hstack([y, y_new_batch])

# Refit the nearest neighbors model
nn_model.fit(train_embeddings)
