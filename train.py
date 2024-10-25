import pandas as pd
import torch
from complEx import ComplEx
from negative_sampler import Negative_sampler
# from utils import AveragePrecision  # Assuming you have an AveragePrecision implementation


def split_train_val(triples, labels, val_ratio=0.1):
    dataset_size = len(triples)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size

    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_triples = triples[train_indices]
    val_triples = triples[val_indices]
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]

    return train_triples, val_triples, train_labels, val_labels


if __name__ == "__main__":
    # Load the CSV
    df = pd.read_csv('books_graph_facts.csv')
    negative_sampler = Negative_sampler()

    # Count unique entities and relations
    unique_relations = df['relation'].unique()
    unique_entities = pd.concat([df['head'], df['tail']]).unique()
    #
    num_entities = len(unique_entities)
    num_relations = len(unique_relations)

    # Create mappings for entities and relations to indices
    entity2id = {entity: idx for idx, entity in enumerate(unique_entities)}
    relation2id = {relation: idx for idx, relation in enumerate(unique_relations)}

    # Convert triples into index-based tensors
    triples = []
    for _, row in df.iterrows():
        subject_id = entity2id[row['head']]
        relation_id = relation2id[row['relation']]
        object_id = entity2id[row['tail']]
        triples.append([subject_id, relation_id, object_id])

    triples = torch.tensor(triples)
    labels = torch.ones(len(triples), dtype=torch.float32)  # Positive labels

    # Generate negative samples
    negatives = negative_sampler.generate_negative_samples(triples, 3*len(triples), num_entities)

    # Combine positive and negative triples and labels
    all_triples = torch.cat([triples, negatives])
    all_labels = torch.cat([torch.ones(len(triples)), torch.zeros(len(negatives))])

    # Split data into training and validation sets
    train_triples, val_triples, train_labels, val_labels = split_train_val(all_triples, all_labels, val_ratio=0.1)


    tensors = {
        "train_triples": train_triples,
        "val_triples": val_triples,
        "train_labels": train_labels,
        "val_labels": val_labels,
        "entity2id": entity2id,
        "relation2id": relation2id
    }

    for name, tensor in tensors.items():
        torch.save(tensor, f'tensor_{name}.pt')

    train_triples = torch.load('tensor_train_triples.pt')
    train_labels = torch.load('tensor_train_labels.pt')
    entity2id = torch.load('tensor_entity2id.pt')
    val_triples = torch.load('tensor_val_triples.pt')
    val_labels = torch.load('tensor_val_labels.pt')

    # Initialize model
    embedding_dim = 512  # Dimensionality of complex vectors
    model = ComplEx(num_entities, num_relations, embedding_dim)

    # Train the model with validation and early stopping
    model.fit(train_triples, train_labels, val_triples, val_labels, num_entities, batch_size=512, epochs=600, lr=0.0005,
              regularization=-1e-7)

    # Save the model's state dictionary
    torch.save(model.state_dict(), f'complex_model_{embedding_dim}.pth')




