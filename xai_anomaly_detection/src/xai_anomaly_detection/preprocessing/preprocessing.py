"""Contains class for data preprocessing"""
import os
from typing import Tuple
import typing
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


class PreprocessNSLKDD:
    """Class for perform different preprocessing steps
    on NSL-KDD data set
    """

    def __init__(self) -> None:
        self.train_data = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "data", "KDDTrain+.txt")
        )
        self.test_data = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "data", "KDDTest+.txt")
        )

        # Get columns name for data
        columns = [
            "duration",
            "protocol_type",
            "service",
            "flag",
            "src_bytes",
            "dst_bytes",
            "land",
            "wrong_fragment",
            "urgent",
            "hot",
            "num_failed_logins",
            "logged_in",
            "num_compromised",
            "root_shell",
            "su_attempted",
            "num_root",
            "num_file_creations",
            "num_shells",
            "num_access_files",
            "num_outbound_cmds",
            "is_host_login",
            "is_guest_login",
            "count",
            "srv_count",
            "serror_rate",
            "srv_serror_rate",
            "rerror_rate",
            "srv_rerror_rate",
            "same_srv_rate",
            "diff_srv_rate",
            "srv_diff_host_rate",
            "dst_host_count",
            "dst_host_srv_count",
            "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate",
            "dst_host_srv_serror_rate",
            "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate",
            "outcome",
            "level",
        ]

        # Assign name for columns
        self.train_data.columns = columns
        self.test_data.columns = columns

    def preprocessing(self) -> None:
        """convert categorical features in binary features with one-hot-encoding
        and normalizes data with min-max normalization
        """

        self.test_data = self._one_hot_encoding(self.test_data)
        self.train_data = self._one_hot_encoding(self.train_data)

        # test data set has fewer unique values in categorical features
        # so we generate the missing feature columns with values '0'
        missing_columns = np.setdiff1d(self.train_data.columns, self.test_data.columns)
        for col in missing_columns:
            self.test_data[col] = 0
        self.test_data = self.test_data.reindex(sorted(self.test_data.columns), axis=1)

        self.test_data = self._min_max_normalization(self.test_data)
        self.train_data = self._min_max_normalization(self.train_data)

        # convert each attack class to class 'attack'
        self.test_data["outcome"].replace(
            to_replace=r"^(?!normal).*$", value="attack", inplace=True, regex=True
        )
        self.train_data["outcome"].replace(
            to_replace=r"^(?!normal).*$", value="attack", inplace=True, regex=True
        )

        # encode labels to integer
        self.test_data["outcome"] = pd.Categorical(self.test_data["outcome"]).codes
        self.train_data["outcome"] = pd.Categorical(self.train_data["outcome"]).codes

    @staticmethod
    def _one_hot_encoding(dataset: pd.DataFrame) -> pd.DataFrame:
        """One hot encoding of every categorical column in dataframe

        Parameters
        ----------
        dataset : pd.DataFrame
            NSL-KDD data set

        Returns
        -------
        pd.DataFrame
            dataframe without categorical columns
        """

        for categorical_column in ["protocol_type", "service", "flag"]:
            # get unique values to create column names
            columns = [
                categorical_column + "_" + x
                for x in dataset[categorical_column].unique()
            ]

            # define one hot encoding
            encoder = OneHotEncoder(sparse=False)
            # transform data
            encoder_df = pd.DataFrame(
                encoder.fit_transform(
                    np.asarray(dataset[categorical_column]).reshape(-1, 1)
                )
            )
            # set column names for one hot encoded data set
            encoder_df.columns = columns

            # merge one hot encoded data into data set
            dataset = dataset.join(encoder_df)

            # drop categorical column
            dataset.drop(categorical_column, axis=1, inplace=True)

        # return sorted dataframe
        return dataset.reindex(sorted(dataset.columns), axis=1)

    @staticmethod
    def _min_max_normalization(dataset: pd.DataFrame) -> pd.DataFrame:
        """Min-Max normalization of dataset

        Parameters
        ----------
        dataset : pd.DataFrame
            data set not normalized

        Returns
        -------
        pd.DataFrame
            normalized dataset in range between 0 and 1
        """

        for colname in dataset.columns:
            if colname == "outcome":
                continue
            dataset[colname] = MinMaxScaler().fit_transform(
                np.asarray(dataset[colname]).reshape(-1, 1)
            )

        return dataset

    def get_data(self, test_data=False, as_df=False) -> typing.Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.DataFrame, pd.Series]]:
        """Returns features and labels of data set in tuple format.
        Labels are encoded to one hot encoded labels

        Parameters
        ----------
        test_data : bool, optional
            whether to return test or train data, by default False

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            features, labels
        """
        if test_data:
            data = self.test_data.copy()
        else:
            data = self.train_data.copy()

        # select the labels into y
        y = data.pop("outcome").values
        # select all features except labels into x
        x = data.values

        return (x, y)
