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
df_data = df_data[df_data["Household Income"].notna()]

df_clean = df_data[["RespondentID"]]
df_clean["RespondentID"] = df_clean["RespondentID"].astype(int).astype(str)


# There are only 5 possible income ranges. no need to parse the strings
possible_income_map = {
    "$0 - $24,999": False,
    "$25,000 - $49,999": False,
    "$50,000 - $99,999": True,
    "$100,000 - $149,999": True,
    "$150,000+": True,
}

df_clean["is_above_50k"] = df_data["Household Income"].map(possible_income_map)

# information about having seen any of the films is redundant to the seen_films columns, so that column won't be considered

# star wars fans either self id'd as fans, self id'd as non-fans, or didn't respond. non-fans and non-respondents are considered false, fans are true.
df_clean["is_fan"] = df_data[
    "Do you consider yourself to be a fan of the Star Wars film franchise?"
].apply(lambda x: True if x == "Yes" else False)


def seen_films(col):
    if pd.isna(col):
        return False
    else:
        return True


df_clean["seen_ep1"] = df_data[
    "Which of the following Star Wars films have you seen? Please select all that apply."
].apply(seen_films)

df_clean["seen_ep2"] = df_data["Unnamed: 4"].apply(seen_films)
df_clean["seen_ep3"] = df_data["Unnamed: 5"].apply(seen_films)
df_clean["seen_ep4"] = df_data["Unnamed: 6"].apply(seen_films)
df_clean["seen_ep5"] = df_data["Unnamed: 7"].apply(seen_films)
df_clean["seen_ep6"] = df_data["Unnamed: 8"].apply(seen_films)

rank_cols = [
    "Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.",
    "Unnamed: 10",
    "Unnamed: 11",
    "Unnamed: 12",
    "Unnamed: 13",
    "Unnamed: 14",
]

# this creates a column that shows only if episode 1 is their top rank.
ep_1_col = rank_cols[0]
ep_1_ranks = df_data[ep_1_col]


def set_ep_1_rank(x):
    if x == 1 or x == "1":
        return "ep_1"
    else:
        return pd.NA


df_clean["fav_episode"] = ep_1_ranks.apply(set_ep_1_rank)

# now I need to iterate through the existing data in "top_ranked_episode" and fill in the rest of the episodes
for i in range(1, len(rank_cols)):

    def rank_value(x):
        if x == 1 or x == "1":
            return f"ep_{i + 1}"
        else:
            return pd.NA

    def fill_ranked_episode():
        target_col = rank_cols[i]
        return df_data[target_col].apply(rank_value)

    df_clean["fav_episode"] = df_clean["fav_episode"].fillna(fill_ranked_episode())

# creating a column for bottom ranked episodes, i can use similar logic to the top rank one
df_clean["least_fav_ep"] = ep_1_ranks.apply(
    lambda x: "ep_1" if x == 6 or x == "6" else pd.NA
)

# here i'll fill in for the rest of the episodes, also same logic as the top rank
for i in range(1, len(rank_cols)):
    df_clean["least_fav_ep"] = df_clean["least_fav_ep"].fillna(
        df_data[rank_cols[i]].apply(
            lambda x: f"ep_{i + 1}" if x == 6 or x == "6" else pd.NA
        )
    )

character_cols_map = {
    "Please state whether you view the following characters favorably, unfavorably, or are unfamiliar with him/her.": "Han",
    "Unnamed: 16": "Luke",
    "Unnamed: 17": "Leia",
    "Unnamed: 18": "Anakin",
    "Unnamed: 19": "Obi_Wan",
    "Unnamed: 20": "Palpatine",
    "Unnamed: 21": "Vader",
    "Unnamed: 22": "Lando",
    "Unnamed: 23": "Boba_Fett",
    "Unnamed: 24": "C-3P0",
    "Unnamed: 25": "R2",
    "Unnamed: 26": "Jar_Jar",
    "Unnamed: 27": "Padme",
    "Unnamed: 28": "Yoda",
}

# the favorability amount can be put on a scale of 1 to 5, so i can reserve 0 for null values
character_response_map = {
    "Unfamiliar (N/A)": pd.NA,
    "Very unfavorably": 1,
    "Somewhat unfavorably": 2,
    "Neither favorably nor unfavorably (neutral)": 3,
    "Somewhat favorably": 4,
    "Very favorably": 5,
}

# I'm treating rows where the record contants "Unfamiliar (N/A)" OR the record just isn't present or isn't one of the other options as pd.NA
for col, name in character_cols_map.items():
    df_clean[name] = df_data[col].apply(lambda x: character_response_map.get(x, pd.NA))

# for rows where the record is pd.NA, or the respondent doesn't understand the question, I'm going to set the value to pd.NA
df_clean["shot_first"] = df_data["Which character shot first?"].apply(
    lambda x: "Han" if x == "Han" else ("Greedo" if x == "Greedo" else pd.NA)
)

# for extended universe I'm going to treat non-fans the same as those who don't know what it is, each of which can be consisdered false
df_clean["eu_fan"] = df_data[
    "Do you consider yourself to be a fan of the Expanded Universe?"
].apply(lambda x: True if x == "Yes" else False)

# star trek fan if simply boolean, non-fans and non-respondents are false
df_clean["is_trekkie"] = df_data[
    "Do you consider yourself to be a fan of the Star Trek franchise?"
].apply(lambda x: True if x == "Yes" else False)

# for purposed of this project, gender is binary, only 2 genders exist in the dataset.
gender_map = {
    "Male": "M",
    "Female": "F",
}

df_clean["sex"] = df_data["Gender"].apply(
    lambda x: pd.NA if pd.isna(x) else gender_map.get(x, pd.NA)
)
# for age, assigning a rank for each range introduces a numerically compariable range so that a larger age ranger can be considered greater than a smaller one.
max_age_map = {
    "18-29": 1,
    "30-44": 2,
    "45-60": 3,
    "> 60": 4,
}

df_clean["age_group"] = df_data["Age"].apply(
    lambda x: pd.NA if pd.isna(x) else max_age_map.get(x, pd.NA)
)

# I'm also ranking the education level, so it can be numerically compared
education_level_map = {
    "Less than high school degree": 1,
    "High school degree": 2,
    "Some college or Associate degree": 3,
    "Bachelor degree": 4,
    "Graduate degree": 5,
}

df_clean["edu_level"] = df_data["Education"].apply(
    lambda x: pd.NA if pd.isna(x) else education_level_map.get(x, pd.NA)
)

census_regions = [
    "Pacific",
    "Mountain",
    "West South Central",
    "West North Central",
    "East South Central",
    "East North Central",
    "South Atlantic",
    "Middle Atlantic",
    "New England",
]

# geographical location isn't logically rankable, so I'm just going to use the string values
df_clean["location"] = df_data["Location (Census Region)"]

pd.set_option("display.max_rows", 100000)
pd.set_option("display.max_columns", 100000)
pd.set_option("display.width", 100000)

df_clean_filtered = df_clean.copy()

df_1_ht = df_clean_filtered.copy()
print(df_1_ht.head(10))

character_favor_chart = ""

shot_first_chart = ""

training_split = [1, 0]
classifier_hyperparameters = {
    "foo": "bar",
}
classifierScore = 0.0

# plan:
# make a binary column for the income threshold to be the target variable
# shoot for the gradient boosting model
# 1 hot encode, but only include top rank & least favorite for the movie rank. also consider dropping a few of the character columns or ranking them instead of 1 hot
# use the ootb 1 hot instead of manual
# gradient booster hyperparameters: max iterations, max depth, node split thingy, learning rate
# test/train split start with 70/30, adjust up and down to see whats best
# find significant features with feature importance
