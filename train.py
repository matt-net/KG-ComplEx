import pandas as pd
from complEx import ComplEx
from negative_sampler import Negative_sampler
from utils import *

if __name__ == "__main__":
    # Load the CSV
    df = pd.read_csv('books_graph_facts.csv')
    negative_sampler = Negative_sampler()


    # Count unique entities and relations
    unique_relations = df['relation'].unique()
    unique_entities = pd.concat([df['head'], df['tail']]).unique()

    num_entities = len(unique_entities)
    num_relations = len(unique_relations)

    # Create mappings for entities and relations to indices
    # entity2id = {entity: idx for idx, entity in enumerate(unique_entities)}
    # relation2id = {relation: idx for idx, relation in enumerate(unique_relations)}

    # # Convert triples into index-based tensors
    # triples = []
    # for _, row in df.iterrows():
    #     subject_id = entity2id[row['head']]
    #     relation_id = relation2id[row['relation']]
    #     object_id = entity2id[row['tail']]
    #     triples.append([subject_id, relation_id, object_id])
    #
    # triples = torch.tensor(triples)
    # labels = torch.ones(len(triples), dtype=torch.float32)  # Positive labels
    #
    # negatives = negative_sampler.generate_negative_samples(triples, len(triples), num_entities)
    #
    # # Generate negative triples and append to training data
    # all_triples = torch.cat([triples, negatives])
    # all_labels = torch.cat([torch.ones(len(triples)), torch.zeros(len(negatives))])
    # tensors = {
    #
    #     "all_triples": all_triples,
    #     "all_labels": all_labels,
    #     "triples": triples,
    #     "entity2id": entity2id,
    #     "relation2id": relation2id
    # }
    #
    # # Save each tensor with a descriptive filename
    # for name, tensor in tensors.items():
    #     torch.save(tensor, f'tensor_{name}.pt')

    all_triples = torch.load('tensor_all_triples.pt')
    all_labels = torch.load('tensor_all_labels.pt')
    triples = torch.load('tensor_triples.pt')
    entity2id = torch.load('tensor_entity2id.pt')
    relation2id = torch.load('tensor_relation2id.pt')

    # Initialize model
    embedding_dim = 256  # Dimensionality of complex vectors
    model = ComplEx(num_entities, num_relations, embedding_dim)

    # Train the model with batching
    model.fit(all_triples, all_labels, num_entities, batch_size=64, epochs=100, lr=0.001, regularization=-10e-7)
    # Save the model's state dictionary
    torch.save(model.state_dict(), f'complex_model_{embedding_dim}.pth')
