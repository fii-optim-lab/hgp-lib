import argparse
import os

import pandas as pd

# Columns describing a user's role/identity within the organization.
USER_COLUMNS = [
    "MGR_ID",
    "ROLE_ROLLUP_1",
    "ROLE_ROLLUP_2",
    "ROLE_DEPTNAME",
    "ROLE_TITLE",
    "ROLE_FAMILY_DESC",
    "ROLE_FAMILY",
    "ROLE_CODE",
]

# Resources for which a per-resource "universal-cross-validation" (UCV) dataset is built.
RESOURCE_LIST = [75078, 25993, 79092, 4675]


def process_aeac(train_csv: str) -> pd.DataFrame:
    """Load train.csv, rename the label column and cast everything to category."""
    print(f"Reading {train_csv}")
    df = pd.read_csv(train_csv)
    df = df.rename(columns={"ACTION": "target"})
    return df.astype("category")


def build_ucv_dataset(df: pd.DataFrame, resource: int) -> pd.DataFrame:
    """
    Build a per-resource dataset augmented with "universal cross-validation" (UCV) rows.

    UCV rows are users that never appear for the given resource. They are added
    as negative (target == 0) examples so the model also learns from users that
    were never granted access to this resource.
    """
    all_users = df[USER_COLUMNS].drop_duplicates().reset_index(drop=True)

    df_for_resource = df[df["RESOURCE"] == resource].reset_index(drop=True)

    all_users_for_resource = all_users.copy()
    all_users_for_resource["target"] = 0
    all_users_for_resource = all_users_for_resource.merge(
        df_for_resource[USER_COLUMNS].drop_duplicates(),
        on=USER_COLUMNS,
        how="left",
        indicator=True,
    )
    all_users_for_resource = all_users_for_resource[
        all_users_for_resource["_merge"] == "left_only"
    ].drop(columns=["_merge"])

    df_for_resource = df_for_resource.drop("RESOURCE", axis=1)

    print(f"Creating UCV dataset for resource {resource}")
    print(f"  {resource} {len(df_for_resource)} All")
    print(f"  {resource} {(df_for_resource['target'] == 0).sum()} -")
    print(f"  {resource} {(df_for_resource['target'] == 1).sum()} +")
    print(f"  {resource} {len(all_users_for_resource)} ucv")

    df_for_resource_ucv = pd.concat([df_for_resource, all_users_for_resource])
    return df_for_resource_ucv.astype("category")


def main():
    parser = argparse.ArgumentParser(
        "AEAC (Amazon Employee Access Challenge) preprocessor"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Datasets folder containing train.csv (outputs are written here)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"'{args.data_path}' is not a directory")

    train_csv = os.path.join(args.data_path, "train.csv")
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(f"'{train_csv}' not found")

    df = process_aeac(train_csv)

    aeac_hdf = os.path.join(args.data_path, "AEAC.hdf")
    print(f"Writing {aeac_hdf}")
    df.to_hdf(aeac_hdf, key="data", mode="w", format="table")

    for resource in RESOURCE_LIST:
        df_for_resource_ucv = build_ucv_dataset(df, resource)
        ucv_hdf = os.path.join(args.data_path, f"AEAC_{resource}_ucv.hdf")
        print(f"Writing {ucv_hdf}")
        df_for_resource_ucv.to_hdf(ucv_hdf, key="data", mode="w", format="table")

    print("Done")


if __name__ == "__main__":
    main()
