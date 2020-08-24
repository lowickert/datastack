import torch
from data_hub.io.resources import StreamedResource
from data_hub.dataset.iterator import DatasetIteratorIF


class MNISTIterator(DatasetIteratorIF):
    """ MNIST dataset iterator (http://yann.lecun.com/exdb/mnist/)
    """

    def __init__(self, samples_stream: StreamedResource, targets_stream: StreamedResource = None):
        dataset_sequences = [torch.load(samples_stream), torch.load(targets_stream)]
        super().__init__(dataset_name="MNIST", dataset_sequences=dataset_sequences)
