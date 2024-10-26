import pandas as pd
from complEx import ComplEx
from negative_sampler import Negative_sampler
from utils import *

if __name__ == "__main__":

    df = pd.read_csv('books_graph_facts.csv')
    negative_sampler = Negative_sampler()

    # Count unique entities and relations
    unique_relations = df['relation'].unique()
    unique_entities = pd.concat([df['head'], df['tail']]).unique()

    num_entities = len(unique_entities)
    num_relations = len(unique_relations)

    '''
    Uncomment these line for generating negative sample and saving them. Then you can comment them again and jump to 
    line 61 for loading them for further analysis. This can save time. (At least it did for me!)
    '''

    # # Create mappings for entities and relations to indices
    # entity2id = {entity: idx for idx, entity in enumerate(unique_entities)}
    # relation2id = {relation: idx for idx, relation in enumerate(unique_relations)}
    #
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
    # # Generate negative samples
    # negatives = negative_sampler.generate_negative_samples(triples, 3*len(triples), num_entities)
    #
    # # Combine positive and negative triples and labels
    # all_triples = torch.cat([triples, negatives])
    # all_labels = torch.cat([torch.ones(len(triples)), torch.zeros(len(negatives))])
    #
    # # Split data into training and validation sets
    # train_triples, val_triples, train_labels, val_labels = split_train_val(all_triples, all_labels, val_ratio=0.1)
    #
    #
    # tensors = {
    #     "train_triples": train_triples,
    #     "val_triples": val_triples,
    #     "train_labels": train_labels,
    #     "val_labels": val_labels,
    #     "entity2id": entity2id,
    #     "relation2id": relation2id
    # }
    #
    # for name, tensor in tensors.items():
    #     torch.save(tensor, f'tensor_{name}.pt')

    train_triples = torch.load('tensor_train_triples.pt')
    train_labels = torch.load('tensor_train_labels.pt')
    entity2id = torch.load('tensor_entity2id.pt')
    val_triples = torch.load('tensor_val_triples.pt')
    val_labels = torch.load('tensor_val_labels.pt')

    embedding_dim = 512
    model = ComplEx(num_entities, num_relations, embedding_dim)

    model.fit(train_triples, train_labels, val_triples, val_labels, num_entities, batch_size=512, epochs=50, lr=0.0001,
              regularization=1e-8)

    torch.save(model.state_dict(), f'complex_model_{embedding_dim}.pth')

    '''
    The following line are used for plotting loss and AP
    '''

    plt.plot(model.epoch_losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()

    plt.plot(model.epoch_ap, label="Validation AP")
    plt.xlabel("Epoch")
    plt.ylabel("Average Precision (AP)")
    plt.title("Validation AP Over Epochs")
    plt.legend()
    plt.show()
