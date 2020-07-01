import csv
import pandas as pd
import os


def read_annotated_file(path, original_file, translation_file, hter_file):

    with open(os.path.join(path,original_file), encoding="utf-8") as f:
        originals = f.read().splitlines()

    with open(os.path.join(path, translation_file), encoding="utf-8") as f:
        translations = f.read().splitlines()

    with open(os.path.join(path, hter_file), encoding="utf-8") as f:
        hters = list(map(float, f.read().splitlines()))

    assert(len(originals) == len(translations))
    assert(len(originals) == len(hters))

    return pd.DataFrame(
                {'original': originals,
                'translation': translations,
                 'hter': hters
                })


def read_test_file(path, original_file, translation_file):

    with open(os.path.join(path, original_file), encoding="utf-8") as f:
        originals = f.read().splitlines()

    with open(os.path.join(path, translation_file), encoding="utf-8") as f:
        translations = f.read().splitlines()

    assert (len(originals) == len(translations))
    indices = list(range(0, len(originals)))

    return pd.DataFrame(
        {'original': originals,
         'translation': translations,
         'index': indices
         })