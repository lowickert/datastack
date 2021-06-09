import pandas as pd
import os

class Dset_converter():

    DSET_LOCATION = "/home/alad/data/fs_data"
    RAW_PATH = "/home/datastack/data/dsets"

    output_names = {
        "train" : {
            "samples" : "fscs5_samples_train.hdf",
            "targets" : "fscs5_targets_train.hdf"
        },
        "test" : {
            "samples" : "fscs5_samples_test.hdf",
            "targets" : "fscs5_targets_test.hdf"
        }
    }

    def __init__(self, input_name="Traffic_Complete_11_cut.hdf"):
        self.input_name = input_name

    def convert_dataset(self):
        # Load dataset
        df = pd.read_hdf(os.path.join(self.DSET_Location, self.input_name))

        # Separate Normal / Anomalous data
        df_normal = df.loc[df["Labels"] == 0]
        df_anomalous = df.loc[df["Labels"] > 0]

        # Create Train / Test set
        df_train = df_normal.sample(frac=0.7)
        df_test = df_normal.loc[~df_normal.index.isin(df_train.index)]
        df_an_train = df_anomalous.sample(frac=0.6)
        df_an_test = df_anomalous.loc[~df_anomalous.index.isin(df_an_train.index)]
        dfs = [df_train, df_test, df_an_train, df_an_test]
        for df in dfs:
            print(df.describe())
        df_train = df_train.append(df_an_train)
        df_test = df_test.appden(df_an_test)

        # Split each set into samples and targets
        datasets = {
            "train" : {
                "samples": df_train.drop("Labels", axis=1),
                "targets": df_train.Labels
            },
            "test": {
                "samples": df_test.drop("Labels", axis=1),
                "targets": df_test.Labels
            }
        }
        
        # Store the resulting four datasets
        for dset in self.output_names.keys():
            for key in dset.keys():
                location = self.output_names[dset][key]
                data = datasets[dset][key]
                print(location)
                print(data.head())