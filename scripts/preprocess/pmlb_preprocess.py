import os
import pmlb
import argparse


def save_pmlb_data(name: str, data_path: str):
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"'{data_path}' is not a directory")
    try:
        data = pmlb.fetch_data(name)
    except ValueError as e:
        raise FileNotFoundError(f"'{name}' not found") from e
    path = os.path.join(data_path, f"{name}.hdf")
    print(f"Writing {path}")
    data.to_hdf(path, key="data", mode="w", format="table")
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("-data", type=str, default="data", help="Datasets folder")
    args = parser.parse_args()

    save_pmlb_data(args.name, args.data)
