from helper_functions import FeatureClean
from helper_functions import FeatureSelect
import pandas as pd
import unittest

# fmt: off
class TestFeatureClean(unittest.TestCase):
    def test_feature_clean(self):
        """
        Test if all features have special characters removed and converted to
        floating or integer data types.
        """
        input_df = pd.DataFrame({
            "x0": [0, 0, 0],
            "x63": ["%5.6", "6.7%", "7.7%"],
            "x12": ["$4,000", "$777", "$(47)"]
        })

        preprocessed_df = FeatureClean().transform(input_df)
        check_0 = preprocessed_df.x63.dtype == "float"
        check_1 = preprocessed_df.x12.dtype == "float"
        self.assertTrue(check_0)
        self.assertTrue(check_1)


class TestFeatureSelect(unittest.TestCase):
    def test_feature_selection(self):
        """
        Test if the feature selection returns a subset of features.
        """
        input_df = pd.DataFrame({
            "x0": [0],
            "x1": [0],
            "x2": [0],
            "x3": [0],
            "x4": [0]
        })

        feature_select = FeatureSelect(
            full_features = input_df.columns.tolist(),
            select_features = ["x0", "x2", "x4"]
        )

        subset_df = feature_select.transform(input_df.to_numpy())
        self.assertTrue(subset_df.shape[1] == 3)


if __name__ == "__main__":
    unittest.main()
