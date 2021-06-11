import pandas as pd
import os
import hashlib

class Dset_converter():

    DSET_LOCATION = "/home/alad/data/fs_data"
    RAW_PATH = "/home/datastack/data/dsets"
    MD5_FILE = "md5_checksums.txt"

    output_names = {
        "train" : {
            "samples" : "fscs5_samples_train.csv",
            "targets" : "fscs5_targets_train.csv"
        },
        "test" : {
            "samples" : "fscs5_samples_test.csv",
            "targets" : "fscs5_targets_test.csv"
        }
    }

    def __init__(self, input_name="Traffic_Complete_11_cut.hdf"):
        self.input_name = input_name

    def convert_dataset(self, verbose=False):
        # Load dataset
        df = pd.read_hdf(os.path.join(self.DSET_LOCATION, self.input_name))
        len_df = len(df)
        df = df.dropna()
        print("{} rows were dropped when dropping NaNs".format(len_df - len(df)))

        # Separate Normal / Anomalous data
        df_normal = df.loc[df["labels"] == 0]
        df_anomalous = df.loc[df["labels"] > 0]

        # Create Train / Test set
        df_train = df_normal.sample(frac=0.7)
        df_test = df_normal.loc[~df_normal.index.isin(df_train.index)]
        df_an_train = df_anomalous.sample(frac=0.6)
        df_an_test = df_anomalous.loc[~df_anomalous.index.isin(df_an_train.index)]
        dfs = [df_train, df_test, df_an_train, df_an_test]
        df_train = df_train.append(df_an_train)
        df_test = df_test.append(df_an_test)

        # Split each set into samples and targets
        datasets = {
            "train" : {
                "samples": df_train.drop("labels", axis=1),
                "targets": df_train.labels
            },
            "test": {
                "samples": df_test.drop("labels", axis=1),
                "targets": df_test.labels
            }
        }

        # Store the resulting four datasets
        md5_file = os.path.join(self.RAW_PATH, self.MD5_FILE)
        print(md5_file)
        for dset in self.output_names.keys():
            print(dset)
            for key in self.output_names[dset].keys():
                location = self.output_names[dset][key]
                data = datasets[dset][key]
                path = os.path.join(self.RAW_PATH, location)
                print(key)
                if verbose:
                    print(os.path.join(self.RAW_PATH, location))
                    print(data.head())
                    print("Length = ", len(data))
                data.to_csv(path)
                print("Stored {}_{} to {}".format(dset, key, path))

                # Calculate md5 of hdf files
                md5 = hashlib.md5()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        md5.update(chunk)
                with open(md5_file, "a+") as f:
                    f.write("{}_{}: {}\n".format(dset, key, md5.hexdigest()))
                print(md5.hexdigest())


if __name__ == "__main__":
    converter = Dset_converter()
    converter.convert_dataset()
