import pandas as pd


class FeatureClean:
    def __init__(self):
        None

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        This function creates a pd.DataFrame containing clean features that have
        all special characters removed e.g. %, $, and parentheses.
        Args:
            input_df: pd.DataFrame of raw input data {x0, x1, ..., x99}
        Returns:
            pd.DataFrame containing the cleaned features.
        """
        output_df = input_df.copy()

        # Remove the money and percentage characters.
        output_df["x12"] = output_df["x12"].str.replace("$", "")
        output_df["x12"] = output_df["x12"].str.replace(",", "")
        output_df["x12"] = output_df["x12"].str.replace(")", "")
        output_df["x12"] = output_df["x12"].str.replace("(", "-")
        output_df["x12"] = output_df["x12"].astype(float)
        output_df["x63"] = output_df["x63"].str.replace("%", "")
        output_df["x63"] = output_df["x63"].astype(float)

        return output_df


# Create a class for feature selection that stores the variable names.
class FeatureSelect:
    def __init__(self, full_features: list, select_features: list):
        self.full_features = full_features
        self.select_features = select_features

    def transform(self, data_in) -> pd.DataFrame:
        """
        Selects the features and returns a subset of columns. The order of
        full_features must align with the column order of data_in.
        Args:
            data_in: array containing feature set.
            full_features: list containing all feature names.
            select_features: list containing feature names to select.
        Returns:
            pd.DataFrame of subset features.
        """
        col_idx = [self.full_features.index(i) for i in self.select_features]
        data_out = pd.DataFrame(
            data_in[:, col_idx], columns=self.select_features
        )
        return data_out


# Create a function that transforms array type back into pd.DataFrame.
def pandas_transform(array_in, pipeline_in) -> pd.DataFrame:
    """
    Transform np.array back to pd.DataFrame with column names.
    Args:
        array_in: np.array containing data.
        pipeline: sklearn pipeline with method .get_feature_names_out()
    Returns:
        pd.DataFrame of data.
    """
    column_names = pipeline_in.get_feature_names_out()
    return pd.DataFrame(array_in, columns=column_names)
