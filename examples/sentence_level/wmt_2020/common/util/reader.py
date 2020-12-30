import csv

import pandas as pd


def read_annotated_file(path, index="index"):
    indices = []
    originals = []
    translations = []
    z_means = []
    with open(path, mode="r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            indices.append(row[index])
            originals.append(row["original"])
            translations.append(row["translation"])
            z_means.append(float(row["z_mean"]))

    return pd.DataFrame(
        {'index': indices,
         'original': originals,
         'translation': translations,
         'z_mean': z_means
         })


def read_test_file(path, index="index"):
    indices = []
    originals = []
    translations = []
    with open(path, mode="r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            indices.append(row[index])
            originals.append(row["original"])
            translations.append(row["translation"])

    return pd.DataFrame(
        {'index': indices,
         'original': originals,
         'translation': translations,
         })
