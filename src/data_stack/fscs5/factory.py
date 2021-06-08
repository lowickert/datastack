#!/usr/bin/env python3

import os
import data_stack
from matplotlib import pyplot as plt
from typing import Tuple, Dict, Any

from data_stack.io.storage_connectors import StorageConnector, FileStorageConnector
from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.io.resource_definition import ResourceDefinition
from data_stack.dataset.iterator import DatasetIteratorIF
from data_stack.dataset.meta import IteratorMeta
from data_stack.io.retriever import RetrieverFactory

from preprocessor import FSCS5Preprocessor
from iterator import FSCS5Iterator




class FSCS5Factory(BaseDatasetFactory):

    def __init__(self, storage_connector: StorageConnector):
        self.raw_path = "fscs5/raw/"
        self.preprocessed_path = "fscs5/preprocessed"
        import_path = os.path.join(os.getcwd(), "data", "dsets")
        self.resource_definitions = {
            "train": [
                ResourceDefinition( identifier=os.path.join(self.raw_path, "fscs5_samples_train.hdf" ),
                                    source=os.path.join(import_path, 'fscs5_samples_train.hdf'),
                                    md5_sum="TODO"
                                    ),
                ResourceDefinition( identifier=os.path.join(self.raw_path, "fscs5_targets_train.hdf"),
                                    source=os.path.join(import_path, "fscs5_targets_train.hdf"),
                                    md5_sum="TODO"
                                    )
            ],
            "test" : [
                ResourceDefinition( identifier=os.path.join(self.raw_path, "fscs5_samples_test.hdf" ),
                                    source=os.path.join(import_path, 'fscs5_samples_test.hdf'),
                                    md5_sum="TODO"
                                    ),
                ResourceDefinition( identifier=os.path.join(self.raw_path, "fscs5_targets_test.hdf"),
                                    source=os.path.join(import_path, "fscs5_targets_test.hdf"),
                                    md5_sum="TODO"
                                    )
            ]
        }

        super().__init__(storage_connector)

    def _get_resource_id(self, data_type: str,  split: str, element: str) -> str:
        return os.path.join("fscs5", data_type, split, element)

    def check_exists(self) -> bool:
        # TODO come up with a better check!
        sample_identifier = self._get_resource_id(data_type="preprocessed", split="train", element="samples.pt")
        return self.storage_connector.has_resource(sample_identifier)

    def _retrieve_raw(self):
        retrieval_jobs =    [ResourceDefinition(identifier=resource_definition.identifier,
                                                source=resource_definition.source,
                                                md5_sum=resource_definition.md5_sum)
                            for split, definitions_list in self.resource_definitions.items()
                            for resource_definition in definitions_list] # Preparing all jobs
        retriever = RetrieverFactory.get_file_retriever(self.storage_connector)
        retriever.retrieve(retrieval_jobs) # Writes to disk, Storage_connector knows where files are stored + how to load them

    def _prepare_split(self, split: str):
        preprocessor = FSCS5Preprocessor(self.storage_connector)
        sample_identifier = self._get_resource_id(data_type="preprocessed", split=split, element="samples.pt")
        target_identifier = self._get_resource_id(data_type="preprocessed", split=split, element="targets.pt")
        preprocessor.preprocess(*[r.identifier for r in self.resource_definitions[split]],  # Split
                                sample_identifier=sample_identifier,                        # Location samples
                                target_identifier=target_identifier)                        # Location targets

    def _get_iterator(self, split: str) -> DatasetIteratorIF:
        splits = self.resource_definitions.keys()
        if split not in splits:
            raise ResourceNotFoundError(f"Split {split} is not defined.")
        if not self.check_exists():
            self._retrieve_raw()
            for s in splits:
                self._prepare_split(s)

    def get_dataset_iterator(self, config: Dict[str, Any] = None) -> Tuple[DatasetIteratorIF, IteratorMeta]:
        return self._get_iterator(**config) # Translates to split : train --> Argument name

if __name__ == "__main__":
    data_stack_root = os.path.join(os.getcwd(), "data")
    fscs5_storage_path = os.path.join(data_stack_root, "cs")
    storage_connector = FileStorageConnector(root_path=fscs5_storage_path)

    fscs5_factory = FSCS5Factory(storage_connector)
    fscs5_iterator = fscs5_factory.get_dataset_iterator(config={"split": "train"})
    img, target = mnist_iterator[0]
    plt.imshow(img)
    plt.show()