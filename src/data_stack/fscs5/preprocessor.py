import io
import torch
import codecs
import numpy as np
import pandas as pd

from data_stack.dataset.preprocesor import PreprocessingHelpers
from data_stack.io.resources import ResourceFactory, StreamedResource
from data_stack.io.storage_connectors import StorageConnector

class FSCS5Preprocessor:

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector

    def preprocess(self, raw_sample_identifier: str, raw_target_identifier: str, sample_identifier: str, target_identifier: str):
        with self._preprocess_sample_resource(raw_sample_identifier, sample_identifier) as sample_resource:
            self.storage_connector.set_resource(identifier=sample_identifier, resource=sample_resource)
            sample_resource.close()
        with self._preprocess_target_resource(raw_target_identifier, target_identifier) as target_resource:
            self.storage_connector.set_resource(identifier=target_identifier, resource=target_resource)
            target_resource.close

    def _torch_tensor_to_streamed_resource(self, identifier: str, tensor: torch.Tensor) -> StreamedResource:
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        resource = ResourceFactory.get_resource(identifier=identifier, file_like_object=buffer)
        return resource # Should be working

    def _preprocess_sample_resource(self, raw_identifier: str, prep_identifier: str) -> StreamedResource:
        with self.storage_connector.get_resource(raw_identifier, ResourceFactory.SupportedStreamedResourceTypes.STREAMED_TEXT_RESOURCE) as raw_resource:
            data = pd.read_hdf(raw_resource)
        resource = self._torch_tensor_to_streamed_resource(prep_identifier, data)
        return resource

    def _preprocess_target_resource(self, raw_identifier: str, prep_identifier: str) -> StreamedResource:
        with self.storage_connector.get_resource(raw_identifier) as raw_resource:
            data = raw_resource.read()
        data = np.frombuffer(data, dtype=np.int64)
        resource = self._torch_tensor_to_streamed_resource(prep_identifier, data)
        return resource
    
    # TODO: Decode Binary Resource to decimal resource to do preprocessing