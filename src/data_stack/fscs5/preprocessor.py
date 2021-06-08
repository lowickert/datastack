from data_stack.io.storage_connectors import StorageConnector

class FSCS5Preprocessor:

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector