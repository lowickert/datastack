import pandas as pd
from data_stack.dataset.iterator import DatasetIterator
from data_stack.io.resources import StreamedResource

class FSCS5Iterator(DatasetIterator):

    def __init__(self, samples_stream: StreamedResource, targets_stream: StreamedResource):
        self.targets = pd.read_csv(targets_stream).drop("datetime", axis=1)
        self.samples = pd.read_csv(samples_stream).drop("datetime", axis=1)
        targets_stream.close()
        samples_stream.close()
        print(self.targets.columns)
        print(self.samples.columns)
    
    def __len__(self):
        return(len(self.samples))

    def __getitem__(self, index:int):
        sample_tensor = self.samples.iloc[index].to_numpy()
        target = self.targets.iloc[index].to_numpy()
        return sample_tensor, target # Is tag and target needed for ATA or is target sufficient?
