import numpy as np
import torch
class Negative_sampler:
    def __init__(self, ):
        self.negative_samples = None



    def corrupt_tail(self, triple, num_entities):
        """
        Corrupt the tail in the given grpah by replacing it with a random entity!
        """
        head, relation, tail = triple

        # Generate a random tail which is not equal to the true tail entity
        corrupted_tail = tail
        while corrupted_tail == tail:
            corrupted_tail = torch.randint(0, num_entities, (1,))

        return head, relation, corrupted_tail

    def generate_negative_samples(self, triples, num_samples, num_entities):

        negative_samples = []
        for i in range(num_samples):
            true_triple = triples[np.random.randint(0, len( triples))]  # Randomly picking a true one
            negative_triple = self.corrupt_tail(true_triple, num_entities)
            negative_samples.append(torch.cat([negative_triple[0].unsqueeze(0),  # head
                                               negative_triple[1].unsqueeze(0),  # relation
                                               negative_triple[2]]))  # tail

        return torch.stack(negative_samples)
