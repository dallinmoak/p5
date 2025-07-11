import pandas as pd
import numpy as np
from lets_plot import *

# from sklearn

LetsPlot.setup_html(isolated_frame=True)


df = pd.read_csv("./StarWars.csv")
df_col_names = df.columns.tolist()

col_2_sample = df[
    "Which of the following Star Wars films have you seen? Please select all that apply."
]


key_row = df.loc[0, :]

df_data = df.drop(index=0)

df_clean = df_data[["RespondentID"]]
df_clean["RespondentID"] = df_clean["RespondentID"].astype(int).astype(str)


def seen_films(col):
    if pd.isna(col):
        return False
    else:
        return True


df_clean["has_seen_ep_1"] = df_data[
    "Which of the following Star Wars films have you seen? Please select all that apply."
].apply(seen_films)

df_clean["has_seen_ep_2"] = df_data["Unnamed: 4"].apply(seen_films)
df_clean["has_seen_ep_3"] = df_data["Unnamed: 5"].apply(seen_films)
df_clean["has_seen_ep_4"] = df_data["Unnamed: 6"].apply(seen_films)
df_clean["has_seen_ep_5"] = df_data["Unnamed: 7"].apply(seen_films)
df_clean["has_seen_ep_6"] = df_data["Unnamed: 8"].apply(seen_films)

df_sample = df_data[
    [
        "Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.",
        "Unnamed: 10",
        "Unnamed: 11",
        "Unnamed: 12",
        "Unnamed: 13",
        "Unnamed: 14",
    ]
].rename(
    columns={
        "Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.": "rate_1",
        "Unnamed: 10": "rate_2",
        "Unnamed: 11": "rate_3",
        "Unnamed: 12": "rate_4",
        "Unnamed: 13": "rate_5",
        "Unnamed: 14": "rate_6",
    }
)

df_clean[""]

pd.set_option("display.max_rows", 100000)
print(df_sample.head(100))
# print(df_clean.head(100))

# plan:
# make a binary column for the income threshold to be the target variable
# shoot for the gradient boosting model
# 1 hot encode, but only include top rank & least favorite for the movie rank. also consider dropping a few of the character columns or ranking them instead of 1 hot
# use the ootb 1 hot instead of manual
# gradient booster hyperparameters: max iterations, max depth, node split thingy, learning rate
# test/train split start with 70/30, adjust up and down to see whats best
# find significant features with feature importance
