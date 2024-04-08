import json
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd


def read_lm_kbc_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """
    Reads a LM-KBC jsonl file and returns a list of dictionaries.
    Args:
        file_path: path to the jsonl file
    Returns:
        list of dictionaries
    """
    with open(file_path, "r") as f:
        rows = [json.loads(line) for line in f]
    return rows


def read_lm_kbc_jsonl_to_df(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Reads a LM-KBC jsonl file and returns a dataframe.
    """
    rows = read_lm_kbc_jsonl(file_path)
    df = pd.DataFrame(rows)
    return df


def save_df_to_jsonl(file_path: Union[str, Path], df : pd.DataFrame):
    """
    Saves the dataframe into a jsonl file.
    """
    with open(file_path, "w") as f:
        for result in df:
            f.write(json.dumps(result) + "\n")