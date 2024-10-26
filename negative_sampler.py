import numpy as np
import torch

class Negative_sampler:
    def __init__(self):
        self.negative_samples = None

    def corrupt_tail(self, triple, num_entities):

        head, relation, tail = triple
        corrupted_tail = tail
        while corrupted_tail == tail:
            corrupted_tail = torch.randint(0, num_entities, (1,)).item()  # Ensure it's a scalar

        return head, relation, corrupted_tail

    def generate_negative_samples(self, triples, num_samples, num_entities):
        true_triples_set = set(tuple(triple) for triple in triples)
        negative_samples = []
        while len(negative_samples) < num_samples:
            true_triple = triples[np.random.randint(0, len(triples))]
            negative_triple = self.corrupt_tail(true_triple, num_entities)

            if tuple(negative_triple) not in true_triples_set:
                negative_samples.append(torch.tensor(negative_triple))

        return torch.stack(negative_samples)
