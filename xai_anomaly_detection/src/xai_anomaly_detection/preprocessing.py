"""Contains class for data proprocessing"""
import os
import pandas as pd

class PreprocessNSLKDD:
    """ Class for perform different preprocessing steps
        on NSL-KDD data set
    """
    def __init__(self) -> None:
        self.train_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'KDDTrain+.txt'))
        self.test_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'KDDTest+.txt'))

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
