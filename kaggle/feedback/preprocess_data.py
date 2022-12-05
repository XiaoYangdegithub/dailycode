import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def create_folds(data, num_splits):
    data["kfold"] = -1

    mskf = MultilabelStratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    labels = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    data_labels = data[labels].values

    for f, (t_, v_) in enumerate(mskf.split(data, data_labels)):
        data.loc[v_, "kfold"] = f

    return data


df = pd.read_csv("train.csv")
train_df = create_folds(df, 5)
train_df.to_csv('train_df_with_fold.csv', index=None)
