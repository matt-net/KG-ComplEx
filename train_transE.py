import pandas as pd
from transE import TransE  # Assuming you saved the TransE class in transE.py
import torch
from utils import *

if __name__ == "__main__":
    # Load the CSV
    df = pd.read_csv('books_graph_facts.csv')

    # Count unique entities and relations
    unique_relations = df['relation'].unique()
    unique_entities = pd.concat([df['head'], df['tail']]).unique()

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

    # Generate negative triples and append to training data
    negative_triples = negative_sampling(triples, num_entities)
    all_triples = torch.cat([triples, negative_triples])

    # Labels: 1 for positive triples, 0 for negative triples
    all_labels = torch.cat([torch.ones(len(triples)), torch.zeros(len(negative_triples))])

    # Initialize model
    embedding_dim = 50  # Dimensionality of entity/relation vectors
    model = TransE(num_entities, num_relations, embedding_dim, margin=1.0)

    # Train the model with batching
    model.fit(all_triples, all_labels, batch_size=64, epochs=150, lr=0.0001, regularization=0.000)

    # Save the model's state dictionary
    torch.save(model.state_dict(), 'transe_model.pth')

    # # Example prediction
    # test_triples = torch.tensor([[entity2id['book_3725'], relation2id['won_award'], entity2id['award_85']]])
    # predictions = model.predict(test_triples)
    # print("Predictions for test triples:", predictions)