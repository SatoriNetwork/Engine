'''
chronos==1.2.1
torch==2.4.1+cpu
'''
from typing import Union
import os
import time
import numpy as np
import torch
from satoriengine.veda.pipelines.interface import PipelineInterface, TrainingResult
from satorilib.logging import debug
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from chronos import ChronosPipeline

class ChronosAdapter:
    def __init__(self, useGPU):
        hfhome = os.environ.get(
            'HF_HOME', default='/Satori/Neuron/models/huggingface')
        os.makedirs(hfhome, exist_ok=True)
        device_map = 'cuda' if useGPU else 'cpu'
        self.device = torch.device(device_map)
        self.model = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-large" if useGPU else "amazon/chronos-t5-small",
            # "amazon/chronos-t5-tiny", # 8M
            # "amazon/chronos-t5-mini", # 20M
            # "amazon/chronos-t5-small", # 46M
            # "amazon/chronos-t5-base", # 200M
            # "amazon/chronos-t5-large", # 710M
            # 'cpu' for any CPU, 'cuda' for Nvidia GPU, 'mps' for Apple Silicon
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            # force_download=True,
        )
        self.ctx_len = 512  # historical context
        self.ctx_len = 512  # historical context
        self.model.model.to(self.device)  # Move model to the appropriate device
    def fit(
        self,
        trainX,
        trainY,
        eval_set=None,
        verbose=False,
        epochs=10,
        batch_size=32,
        learning_rate=1e-4,
    ):
        # Convert training data to PyTorch tensors
        trainX_tensor = torch.tensor(
            trainX,
            dtype=torch.float32
        ).to(self.device)
        trainY_tensor = torch.tensor(
            trainY,
            dtype=torch.float32
        ).to(self.device)
        # Create a DataLoader for batching
        train_data = TensorDataset(trainX_tensor, trainY_tensor)
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4)
        # Define optimizer and loss function
        optimizer = optim.Adam(
            self.model.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5)
        loss_fn = nn.MSELoss()  # Assuming regression; you can change this if needed
        # Training loop
        for epoch in range(epochs):
            self.model.model.train()  # Set model to training mode
            epoch_loss = 0
            for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
                # Clear gradients
                optimizer.zero_grad()
                # Forward pass
                # Use the last `ctx_len` as context
                context = X_batch[:, -self.ctx_len:]
                context = torch.unsqueeze(context, -1)  # Add channel dimension
                context = context.squeeze(-1).long()  # Convert to torch.LongTensor,
                attention_mask = torch.ones_like(context, dtype=torch.long).to(self.device)
                forecast = self.model.model(
                    context,
                    attention_mask=attention_mask,
                    prediction_length=1,
                    num_samples=4)
                # Compute loss
                predictions = forecast.median(dim=1).values
                assert predictions.requires_grad, "Predictions do not require gradients"
                # Compute loss
                Y_batch = Y_batch.to(self.device)
                loss = loss_fn(predictions, Y_batch)
                assert loss.requires_grad, "Loss does not require gradients"
                epoch_loss += loss.item()
                # Backpropagation and optimization
                loss.backward()
                optimizer.step()
            avg_loss = epoch_loss / len(train_loader)
            if verbose:
                debug(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
            # Optionally: Evaluate on eval_set
            if eval_set is not None:
                self.evaluate(eval_set)
        return TrainingResult(1, self, False)

def generate_training_data(
    num_samples=1000,
    input_dim=512,
    output_dim=1,
    seed=42
):
    """
    Generate synthetic training data for testing the fit function.
    Args:
        num_samples (int): Number of samples in the dataset.
        input_dim (int): Number of features in each input sample.
        output_dim (int): Number of outputs for each target sample.
        seed (int): Random seed for reproducibility.
    Returns:
        tuple: (trainX, trainY) where trainX is the input data and trainY is the target data.
    """
    np.random.seed(seed)
    # Generate random input data
    trainX = np.random.rand(num_samples, input_dim).astype(np.float32)
    # Generate target data with some simple relationship to inputs
    trainY = np.sum(trainX, axis=1, keepdims=True).astype(np.float32)  # Sum of inputs as target
    return trainX, trainY

trainX, trainY = generate_training_data(num_samples=1000, input_dim=512, output_dim=1)

c = ChronosAdapter(useGPU=False)
c.fit(trainX, trainY, epochs=10, verbose=True)
