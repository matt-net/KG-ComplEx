import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from negative_sampler import Negative_sampler
from sklearn.metrics import average_precision_score

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
        nn.init.normal_(self.entity_real.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.entity_imag.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.relation_real.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.relation_imag.weight, mean=0.0, std=0.1)

    def forward(self, s, r, o):
        s_real = self.entity_real(s)
        s_imag = self.entity_imag(s)
        r_real = self.relation_real(r)
        r_imag = self.relation_imag(r)
        o_real = self.entity_real(o)
        o_imag = self.entity_imag(o)

        # Compute the ComplEx score ( Page 11 of the paper)
        score_ = torch.sum(s_real * r_real * o_real + s_imag * r_real * o_imag, dim=1) + torch.sum(
            s_real * r_imag * o_imag - s_imag * r_imag * o_real, dim=1)
        return score_

    def fit(self, triples, labels, val_triples, val_labels, num_entities, batch_size=64, epochs=100, lr=0.0001,
            regularization=0.0, patience=10):
        self.to(self.device)

        # Create lists to store loss and AP for each epoch
        epoch_losses = []
        epoch_ap = []

        dataset = TensorDataset(triples, labels)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        best_ap = 0
        patience_counter = 0

        for epoch in range(epochs):
            total_loss = 0.0
            self.train()

            for batch_triples, batch_labels in data_loader:
                optimizer.zero_grad()

                batch_triples, batch_labels = batch_triples.to(self.device), batch_labels.to(self.device)

                subjects = batch_triples[:, 0]
                relations = batch_triples[:, 1]
                objects = batch_triples[:, 2]

                scores = self.forward(subjects, relations, objects)
                loss = criterion(scores, batch_labels)

                if regularization > 0.0:
                    l2_loss = (self.entity_real.weight.norm(2) ** 2 +
                               self.entity_imag.weight.norm(2) ** 2 +
                               self.relation_real.weight.norm(2) ** 2 +
                               self.relation_imag.weight.norm(2) ** 2)
                    loss += 2 * regularization * l2_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Store average loss for this epoch
            avg_loss = total_loss / len(data_loader)
            epoch_losses.append(avg_loss)

            # Calculate and store AP on validation set
            val_ap = self.compute_average_precision(val_triples, val_labels, batch_size)
            epoch_ap.append(val_ap)

            print(f"Epoch {epoch}, Loss: {avg_loss}, Val AP: {val_ap}")

            # Check for early stopping based on AP
            if val_ap > best_ap:
                best_ap = val_ap
                patience_counter = 0  # Reset the patience counter
                print(f"New best AP: {best_ap}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Save losses and APs for later plotting
        self.epoch_losses = epoch_losses
        self.epoch_ap = epoch_ap

    def predict(self, triples):
        self.to(self.device)
        triples = triples.to(self.device)

        subjects = triples[:, 0]
        relations = triples[:, 1]
        objects = triples[:, 2]
        scores = self.forward(subjects, relations, objects)
        return torch.sigmoid(scores)

    def hits_at_k(self, triples, labels, batch_size, k):
        self.to(self.device)

        dataset = TensorDataset(triples, labels)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        hits = 0
        total_pos_samples = 0

        self.eval()  # Set model to evaluation mode

        with torch.no_grad():
            for batch_triples, batch_labels in data_loader:
                batch_triples = batch_triples.to(self.device)
                batch_labels = batch_labels.to(self.device)

                scores = self.predict(batch_triples)

                pos_mask = (batch_labels == 1)
                neg_mask = (batch_labels == 0)

                pos_scores = scores[pos_mask]
                neg_scores = scores[neg_mask]

                if len(pos_scores) == 0:
                    continue  # No positive samples in this batch :( (Bad luck!)

                all_scores = torch.cat([pos_scores, neg_scores], dim=0)

                _, sorted_indices = torch.sort(all_scores, descending=True)

                sorted_pos_ranks = torch.nonzero(sorted_indices < len(pos_scores)).view(
                    -1) + 1

                hits_in_top_k = (sorted_pos_ranks <= k).sum().item()

                hits += hits_in_top_k
                total_pos_samples += len(pos_scores)

        hits_at_k = hits / total_pos_samples if total_pos_samples > 0 else 0
        return hits_at_k

    def compute_average_precision(self, val_triples, val_labels, batch_size):
        self.eval()
        all_scores = []
        all_labels = []

        val_dataset = TensorDataset(val_triples, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch_triples, batch_labels in val_loader:
                batch_triples = batch_triples.to(self.device)
                batch_labels = batch_labels.to(self.device)

                subjects = batch_triples[:, 0]
                relations = batch_triples[:, 1]
                objects = batch_triples[:, 2]
                scores = self.forward(subjects, relations, objects)
                all_scores.append(scores.cpu())
                all_labels.append(batch_labels.cpu())

        all_scores = torch.cat(all_scores).numpy()
        all_labels = torch.cat(all_labels).numpy()
        ap = average_precision_score(all_labels, all_scores)
        return ap
