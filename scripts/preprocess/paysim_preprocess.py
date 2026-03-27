import argparse
import os.path

import pandas as pd


def process_paysim(path):
    dtype = {
        "type": "category",
        "amount": "float32",
        "oldbalanceOrg": "float32",
        "newbalanceOrig": "float32",
        "oldbalanceDest": "float32",
        "newbalanceDest": "float32",
        "isFraud": "bool",
    }

    print("Reading PaySim.csv")
    df = pd.read_csv(path, usecols=tuple(dtype.keys()), dtype=dtype)
    print("Processing PaySim.csv")
    df.rename(columns={"isFraud": "target"}, inplace=True)
    df["externalOrig"] = (df["oldbalanceOrg"] == 0.0) & (df["newbalanceOrig"] == 0.0)
    df["externalDest"] = (df["oldbalanceDest"] == 0.0) & (df["newbalanceDest"] == 0.0)
    return df


def main():
    parser = argparse.ArgumentParser("PaySim preprocessor")
    parser.add_argument("-data", type=str, default="data", help="Datasets folder")
    args = parser.parse_args()
    if not os.path.isdir(args.data):
        raise FileNotFoundError(f"'{args.data}' is not a directory")
    paysim_csv = os.path.join(args.data, "PaySim.csv")
    if not os.path.isfile(paysim_csv):
        raise FileNotFoundError(f"'{paysim_csv}' not found")

    df = process_paysim(paysim_csv)
    print("Writing PaySim.hdf")
    df.to_hdf(
        os.path.join(args.data, "PaySim.hdf"), key="data", mode="w", format="table"
    )
    print("Done")


if __name__ == "__main__":
    main()
