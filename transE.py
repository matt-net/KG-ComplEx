import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0):
        super(TransE, self).__init__()

        # Entity and relation embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Initialize embeddings with uniform distribution
        nn.init.uniform_(self.entity_embeddings.weight, a=-6/(embedding_dim**0.5), b=6/(embedding_dim**0.5))
        nn.init.uniform_(self.relation_embeddings.weight, a=-6/(embedding_dim**0.5), b=6/(embedding_dim**0.5))

        self.margin = margin
        self.criterion = nn.MarginRankingLoss(margin=self.margin)  # Margin-based loss for link prediction

    def forward(self, s, r, o):
        # Get embeddings for subject, relation, object
        s_emb = self.entity_embeddings(s)
        r_emb = self.relation_embeddings(r)
        o_emb = self.entity_embeddings(o)

        # Compute TransE score (L2 distance)
        score = torch.norm(s_emb + r_emb - o_emb, p=2, dim=1)
        return score

    def fit(self, triples, labels, batch_size=64, epochs=100, lr=0.0001, regularization=0.0):
        # Create DataLoader for batching
        dataset = TensorDataset(triples, labels)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=lr)  # Adam optimizer

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_triples, batch_labels in data_loader:
                optimizer.zero_grad()

                # Extract subject, relation, object from the batch
                subjects = batch_triples[:, 0]
                relations = batch_triples[:, 1]
                objects = batch_triples[:, 2]

                # Forward pass
                scores = self.forward(subjects, relations, objects)

                # Prepare labels for margin ranking loss
                positive_scores = scores[batch_labels == 1]  # Positive scores
                negative_scores = scores[batch_labels == 0]

                target = torch.ones(positive_scores.size(), device=scores.device)  # Target for ranking loss
                loss = self.criterion(positive_scores, negative_scores, target)

                # Add L2 regularization (optional)
                if regularization > 0.0:
                    l2_loss = (self.entity_embeddings.weight.norm(2) ** 2 +
                               self.relation_embeddings.weight.norm(2) ** 2)
                    loss += 2 * regularization * l2_loss

                # Backpropagation
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 1 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(data_loader)}")

        print("Training complete!")

    def predict(self, triples):
        subjects = triples[:, 0]
        relations = triples[:, 1]
        objects = triples[:, 2]
        scores = self.forward(subjects, relations, objects)
        return scores  # Return distances for link prediction