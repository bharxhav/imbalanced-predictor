"""
This module splits the dataset into expanded training sets.
"""
import pandas as pd


class Preprocessor:
    """
    Preprocesses the dataset by splitting it into expanded training sets.
    """

    def __init__(self, dataframe: pd.DataFrame, target_feature: str) -> None:
        self.df = dataframe
        self.target_feature = target_feature
        self.features = {}

    def _binary_encode_positives(self) -> None:
        """
        Binary encodes the positive class, for each unique value in target_feature.
        """
        unique_values = self.df[self.target_feature].unique()

        for class_name in unique_values:
            indicator_series = (
                self.df[self.target_feature] == class_name).astype(int)

            self.features[class_name] = indicator_series

    def get_indicators(self) -> pd.DataFrame:
        """
        Returns the master indicators.
        """
        if not self.features:
            self._binary_encode_positives()

        return pd.DataFrame(self.features)
