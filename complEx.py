import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from negative_sampler import Negative_sampler


class ComplEx(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(ComplEx, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using {self.device}")
        self.batch_size = 64
        # Real and imaginary parts for entity embeddings
        self.entity_real = nn.Embedding(num_entities, embedding_dim)
        self.entity_imag = nn.Embedding(num_entities, embedding_dim)

        self.negative_sampler = Negative_sampler()

        # Real and imaginary parts for relation embeddings
        self.relation_real = nn.Embedding(num_relations, embedding_dim)
        self.relation_imag = nn.Embedding(num_relations, embedding_dim)

        # Initialize embeddings with normal dist.
        nn.init.normal_(self.entity_real.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.entity_imag.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.relation_real.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.relation_imag.weight, mean=0.0, std=1.0)

    def forward(self, s, r, o):
        # Get real and imaginary parts of subject, relation, object
        s_real = self.entity_real(s)
        s_imag = self.entity_imag(s)
        r_real = self.relation_real(r)
        r_imag = self.relation_imag(r)
        o_real = self.entity_real(o)
        o_imag = self.entity_imag(o)

        # Compute the ComplEx score ( Page 11 of the paper)
        score_ = torch.sum(s_real * r_real * o_real + s_imag * r_real * o_imag, dim=1) + torch.sum(
            s_real * r_imag * o_imag - s_imag * r_imag * o_real, dim=1)
        return torch.sigmoid(score_)

    def fit(self, triples, labels, num_entities, batch_size=64, epochs=100, lr=0.0001, regularization=0.0):
        # Move model to the available device (GPU or CPU)
        self.to(self.device)

        # Create DataLoader for batching
        dataset = TensorDataset(triples, labels)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy for link prediction
        optimizer = optim.Adam(self.parameters(), lr=lr)  # Adam optimizer

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_triples, batch_labels in data_loader:
                optimizer.zero_grad()

                # negatives = self.negative_sampler.generate_negative_samples(batch_triples, batch_size, num_entities)
                # negatives = negatives.to(self.device)
                # Move batch data to the available device (GPU or CPU)
                batch_triples, batch_labels = batch_triples.to(self.device), batch_labels.to(self.device)
                # batch_triples = torch.cat([batch_triples, negatives])
                # batch_labels = torch.cat([batch_labels, torch.zeros(len(negatives), device=self.device)])

                # Extract subject, relation, object from the batch
                subjects = batch_triples[:, 0]
                relations = batch_triples[:, 1]
                objects = batch_triples[:, 2]

                # Forward pass
                scores = self.forward(subjects, relations, objects)

                # Compute loss
                loss = criterion(scores, batch_labels)

                # Add L2 regularization (optional)
                if regularization > 0.0:
                    l2_loss = (self.entity_real.weight.norm(2) ** 2 +
                               self.entity_imag.weight.norm(2) ** 2 +
                               self.relation_real.weight.norm(2) ** 2 +
                               self.relation_imag.weight.norm(2) ** 2)
                    loss += 2 * regularization * l2_loss

                # Backpropagation
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 1 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(data_loader)}")

        print("Training complete!")

    def predict(self, triples):
        self.to(self.device)
        triples = triples.to(self.device)

        subjects = triples[:, 0]
        relations = triples[:, 1]
        objects = triples[:, 2]
        scores = self.forward(subjects, relations, objects)
        return scores

    def hits_at_k(self, triples, labels, batch_size, k):
        self.to(self.device)

        # Create DataLoader for batching
        dataset = TensorDataset(triples, labels)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        hits = 0
        total_pos_samples = 0

        self.eval()  # Set model to evaluation mode

        with torch.no_grad():
            for batch_triples, batch_labels in data_loader:
                batch_triples = batch_triples.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Get the model predictions (e.g., scores or logits)
                scores = self.predict(batch_triples)

                # Separate positive and negative samples
                pos_mask = (batch_labels == 1)
                neg_mask = (batch_labels == 0)

                # Get ranks for positive samples by comparing against all predictions
                pos_scores = scores[pos_mask]
                neg_scores = scores[neg_mask]

                if len(pos_scores) == 0:
                    continue  # No positive samples in this batch

                # Concatenate the positive and negative scores for ranking
                all_scores = torch.cat([pos_scores, neg_scores], dim=0)

                # Sort the scores in descending order (higher scores = higher ranks)
                _, sorted_indices = torch.sort(all_scores, descending=True)

                # Identify positions of positive samples in the sorted ranking
                sorted_pos_ranks = torch.nonzero(sorted_indices < len(pos_scores)).view(
                    -1) + 1  # Convert 0-based to 1-based

                # Count the positive samples that are ranked within the top-k
                hits_in_top_k = (sorted_pos_ranks <= k).sum().item()

                hits += hits_in_top_k
                total_pos_samples += len(pos_scores)

        hits_at_k = hits / total_pos_samples if total_pos_samples > 0 else 0
        return hits_at_k
