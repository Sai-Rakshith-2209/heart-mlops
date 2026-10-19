import numpy as np
import pandas as pd

from heart import config
from heart.data import clean, parse_mirror, parse_uci

UCI_ROWS = (
    b"63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0\n"
    b"67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2\n"
    b"57.0,0.0,4.0,140.0,241.0,0.0,0.0,123.0,1.0,0.2,2.0,?,7.0,1\n"
)


def test_parse_uci_marks_question_marks_missing():
    df = parse_uci(UCI_ROWS)
    assert list(df.columns) == config.COLUMNS
    assert df["ca"].isna().sum() == 1


def test_parse_mirror_maps_strings_to_uci_codes():
    raw = (
        b",Age,Sex,ChestPain,RestBP,Chol,Fbs,RestECG,MaxHR,ExAng,Oldpeak,Slope,Ca,Thal,AHD\n"
        b"1,63,1,typical,145,233,1,2,150,0,2.3,3,0,fixed,No\n"
        b"2,67,1,asymptomatic,160,286,0,2,108,1,1.5,2,NA,reversable,Yes\n"
    )
    df = parse_mirror(raw)
    assert df["cp"].tolist() == [1, 4]
    assert df["thal"].tolist() == [6, 7]
    assert df["num"].tolist() == [0, 1]
    assert np.isnan(df.loc[1, "ca"])


def test_clean_binarises_target_and_keeps_missing_for_pipeline():
    out = clean(parse_uci(UCI_ROWS))
    assert out[config.TARGET].tolist() == [0, 1, 1]
    assert "num" not in out.columns
    assert out["ca"].isna().sum() == 1  # imputed later, inside the pipeline


def test_clean_nulls_out_of_domain_values_and_drops_duplicates():
    df = parse_uci(UCI_ROWS + UCI_ROWS[:66])  # duplicate first row
    df.loc[1, "cp"] = 9  # invalid category
    df.loc[2, "chol"] = 5000  # impossible value
    out = clean(df)
    assert len(out) == 3  # row 3 was an exact duplicate of row 0
    assert pd.isna(out.loc[1, "cp"]) and pd.isna(out.loc[2, "chol"])


def test_clean_dataset_contract(clean_df):
    assert len(clean_df) == 303
    assert set(clean_df[config.TARGET].unique()) == {0, 1}
    assert set(config.FEATURES).issubset(clean_df.columns)
    assert clean_df[config.FEATURES].isna().sum().sum() <= 10
