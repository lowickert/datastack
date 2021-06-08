import torch
from data_stack.dataset.iterator import SequenceDatasetIterator
from data_stack.io.resources import StreamedResource

class FSCS5Iterator(SequenceDatasetIterator):

    def __init__(self, samples_stream: StreamedResource, targets_stream: StreamedResource):
        targets = [int(target) for target in torch.load(targets_stream)]
        dataset_sequences = [torch.load(samples_stream), targets] # Label only loaded just once
        samples_stream.close()
        super().__init__(dataset_sequences=dataset_sequences)