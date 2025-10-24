import pandas as pd


def binarize(data: pd.DataFrame):
    if not isinstance(data, pd.DataFrame):
        # TODO: Add test for this
        raise TypeError(f"Expected data to be pandas DataFrame, got {type(data)}")
    #
    # binarized_data = {}
    # for column in data.columns:
    #     match