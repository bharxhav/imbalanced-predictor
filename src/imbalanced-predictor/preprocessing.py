"""
This module splits the dataset into expanded training sets.
"""
import pandas as pd


class Preprocessor:
    """
    Preprocesses the dataset by splitting it into expanded training sets.
    """

    def __init__(self, dataframe: pd.DataFrame, target_class: str) -> None:
        self.df = dataframe
        self.target_class = target_class

    def _categorize_data(self) -> dict:
        """
        Categorizes the data into dictionary of classes.
        """

        features = {}

        for class_name in self.df[self.target_class].unique():
            features[class_name] = self.df[self.df[self.target_class] == class_name]

        return features
