import random
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

import random
import torch


def negative_sampling(triples, num_entities, negative_fraction=0.1):
    # Convert the tensor to a list for random.choice
    triples_list = triples.tolist()

    num_positive_samples = len(triples_list)
    num_negative_samples = int(num_positive_samples * negative_fraction)

    negative_triples = []

    while len(negative_triples) < num_negative_samples:
        # Randomly choose a positive triple from the list
        s, r, o = random.choice(triples_list)
        corrupt_head = random.random() > 0.5

        if corrupt_head:
            s_neg = random.randint(0, num_entities - 1)  # Replace head
            negative_triple = [s_neg, r, o]
        else:
            o_neg = random.randint(0, num_entities - 1)  # Replace tail
            negative_triple = [s, r, o_neg]

        # Ensure the negative triple is unique
        if negative_triple not in triples_list and negative_triple not in negative_triples:
            negative_triples.append(negative_triple)

    return torch.tensor(negative_triples)
#


def negative_sampling(triples, num_entities):
    negative_triples = []
    for s, r, o in triples:
        corrupt_head = random.random() > 0.5
        if corrupt_head:
            s_neg = random.randint(0, num_entities - 1)  # Replace head
            negative_triples.append([s_neg, r, o])
        else:
            o_neg = random.randint(0, num_entities - 1)  # Replace tail
            negative_triples.append([s, r, o_neg])
    return torch.tensor(negative_triples)


import numpy as np


def filtered_mrr(model, test_triples, num_entities):
    """
    Calculate Filtered Mean Reciprocal Rank (MRR).

    Parameters:
    - model: Trained ComplEx model
    - test_triples: Tensor of shape (num_tests, 3) containing subject, relation, and object indices
    - num_entities: Total number of entities in the dataset

    Returns:
    - MRR: Filtered Mean Reciprocal Rank
    """
    ranks = []

    for triple in test_triples:
        subject, relation, true_object = triple

        # Get scores for all entities
        subjects_tensor = torch.tensor([subject] * num_entities)  # Repeat the subject for all entities
        relations_tensor = torch.tensor([relation] * num_entities)  # Repeat the relation for all entities
        all_entities = torch.arange(num_entities)  # All possible objects

        with torch.no_grad():  # No need to track gradients
            scores = model.predict(torch.stack((subjects_tensor, relations_tensor, all_entities), dim=1))

        # Create a ranking based on scores (highest score first)
        sorted_indices = torch.argsort(scores, descending=True)

        # Find the rank of the true object
        rank = (sorted_indices == true_object).nonzero(as_tuple=True)[0].item() + 1  # +1 for 1-based rank
        ranks.append(rank)

    # Calculate MRR
    mrr = np.mean(1.0 / np.array(ranks))
    return mrr


# Reduce the dimensions using t-SNE
def reduce_dimensions_tsne(embeddings, n_components=2, random_state=42):
    tsne = TSNE(n_components=n_components, random_state=random_state)
    reduced_embeddings = tsne.fit_transform(embeddings)
    return reduced_embeddings


# Function to extract real and imaginary parts of entity embeddings
def get_entity_embeddings(model):
    entity_real = model.entity_real.weight.detach().cpu().numpy()
    entity_imag = model.entity_imag.weight.detach().cpu().numpy()
    return entity_real, entity_imag


# Function to visualize the embeddings with labels
def plot_with_labels(embeddings, labels, title):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], palette="deep", legend="full", s=50)
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Labels")
    plt.show()

