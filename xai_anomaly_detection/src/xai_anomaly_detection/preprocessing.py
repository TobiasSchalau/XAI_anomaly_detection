"""Contains class for data proprocessing"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class PreprocessNSLKDD:
    """ Class for perform different preprocessing steps
        on NSL-KDD data set
    """

    def __init__(self) -> None:
        self.train_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'KDDTrain+.txt'))
        self.test_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'KDDTest+.txt'))

        # Get columns name for data
        columns = (['duration'
        ,'protocol_type'
        ,'service'
        ,'flag'
        ,'src_bytes'
        ,'dst_bytes'
        ,'land'
        ,'wrong_fragment'
        ,'urgent'
        ,'hot'
        ,'num_failed_logins'
        ,'logged_in'
        ,'num_compromised'
        ,'root_shell'
        ,'su_attempted'
        ,'num_root'
        ,'num_file_creations'
        ,'num_shells'
        ,'num_access_files'
        ,'num_outbound_cmds'
        ,'is_host_login'
        ,'is_guest_login'
        ,'count'
        ,'srv_count'
        ,'serror_rate'
        ,'srv_serror_rate'
        ,'rerror_rate'
        ,'srv_rerror_rate'
        ,'same_srv_rate'
        ,'diff_srv_rate'
        ,'srv_diff_host_rate'
        ,'dst_host_count'
        ,'dst_host_srv_count'
        ,'dst_host_same_srv_rate'
        ,'dst_host_diff_srv_rate'
        ,'dst_host_same_src_port_rate'
        ,'dst_host_srv_diff_host_rate'
        ,'dst_host_serror_rate'
        ,'dst_host_srv_serror_rate'
        ,'dst_host_rerror_rate'
        ,'dst_host_srv_rerror_rate'
        ,'outcome'
        ,'level'])

        # Assign name for columns
        self.train_data.columns = columns
        self.test_data.columns  = columns

    def preprocessing(self) -> None:
        """ convert categorial features in binary features with one-hot-encoding
            and normalizes data with min-max normalization
        """

        self.test_data = self._one_hot_encoding(self.test_data)
        self.train_data = self._one_hot_encoding(self.train_data)

        # test data set has fewer unique values in categorial features
        # so we generate the missing feature columns with values '0'
        missing_columns = np.setdiff1d(self.train_data.columns, self.test_data.columns)
        for col in missing_columns:
            self.test_data[col] = 0
        self.test_data = self.test_data.reindex(sorted(self.test_data.columns), axis=1)

    @staticmethod
    def _one_hot_encoding(dataset: pd.DataFrame) -> pd.DataFrame:
        """One hot encoding of every categorial column in dataframe

        Parameters
        ----------
        dataset : pd.DataFrame
            NSL-KDD data set

        Returns
        -------
        pd.DataFrame
            dataframe without categorial columns
        """

        for categorial_column in ['protocol_type', 'service', 'flag']:
            # get unique values to create column names
            columns = [categorial_column+'_'+x for x in dataset[categorial_column].unique()]

            # define one hot encoding
            encoder = OneHotEncoder(sparse=False)
            # transform data
            encoder_df = pd.DataFrame(encoder.fit_transform(np.asarray(dataset[categorial_column]).reshape(-1,1)))
            # set column names for on ehot encoded data set
            encoder_df.columns = columns

            # merge one hot encoded data into data set
            dataset = dataset.join(encoder_df)

            # drop categorial column
            dataset.drop(categorial_column, axis=1, inplace=True)

        # return sorted dataframe
        return dataset.reindex(sorted(dataset.columns), axis=1)

    def print_head(self, dataset: str) -> None:
        """ Prints head of given dataset

        Parameters
        ----------
        dataset : str
            'train' or 'test' data set
        """
        if dataset=='train':
            print('Train data set:')
            print(self.train_data.head())
        elif dataset=='test':
            print('Test data set:')
            print(self.test_data.head())
        else:
            print('Dataset does not exist')
