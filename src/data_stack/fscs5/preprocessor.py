import io
import pandas as pd

from data_stack.dataset.preprocesor import PreprocessingHelpers
from data_stack.io.resources import ResourceFactory, StreamedResource
from data_stack.io.storage_connectors import StorageConnector

class FSCS5Preprocessor:

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector

    def preprocess(self, raw_sample_identifier: str, raw_target_identifier: str, sample_identifier: str, target_identifier: str):
        sample_df = self._preprocess_sample_resource(raw_sample_identifier, sample_identifier) 
        target_df = self._preprocess_target_resource(raw_target_identifier, target_identifier) 
        sample_resource = self._df_to_streamed_recourse(sample_identifier, sample_df)
        target_resource = self._df_to_streamed_recourse(target_identifier, target_df)
        self.storage_connector.set_resource(sample_identifier, sample_resource)
        self.storage_connector.set_resource(target_identifier, target_resource)

    def _preprocess_sample_resource(self, raw_identifier: str, prep_identifier: str) -> StreamedResource:
        with self.storage_connector.get_resource(raw_identifier, ResourceFactory.SupportedStreamedResourceTypes.STREAMED_TEXT_RESOURCE) as raw_resource:
            data = pd.read_csv(raw_resource)
        # Insert preprocessing if necessary
        return data

    def _preprocess_target_resource(self, raw_identifier: str, prep_identifier: str) -> StreamedResource:
        with self.storage_connector.get_resource(raw_identifier) as raw_resource:
            data = pd.read_csv(raw_resource)
        # Insert preprocessing if necessary
        return data

    def _df_to_streamed_recourse(self, identifier: str, df: pd.DataFrame) -> StreamedResource:
        string_buffer = io.StringIO()
        df.to_csv(path_or_buf=string_buffer, index=False)
        string_buffer.seek(0)
        byte_buffer = io.BytesIO(string_buffer.read().encode('utf8'))
        resource = ResourceFactory.get_resource(identifier=identifier, file_like_object=byte_buffer)
        return resource