# %%
import pandas as pd
import numpy as np
import matplotlib as plt

# %%
df=pd.read_csv('/Users/moidris/Downloads/footy3.csv')

# %%
df

# %%
import re

def clean_special_characters(name):
    # Remove special characters, except accents, hyphen, apostrophe, and caret
    cleaned_name = re.sub(r'[^\w\s\'\-\^À-ÖØ-öø-ÿ]', '', name)
    
    # Capitalize the first letter of each word
    cleaned_name = ' '.join(word.capitalize() for word in cleaned_name.split())
    
    return cleaned_name

df['Player'] = df['Player'].apply(clean_special_characters)

# %%
import re
from IPython.display import display, HTML

special_characters_pattern = re.compile(r'[^A-Za-z0-9\s]')
players_with_special_characters = df[df['Player'].str.contains(special_characters_pattern)]

# Display players with special characters in a scrollable HTML element
#display(HTML(players_with_special_characters[['Player']].to_html()))

print(players_with_special_characters['Player'])

# %%
df['Player_League_Club'] = df['Player'] + ' - ' + df['Comp'] + ' - ' + df['Squad']

# %%
# Drop the individual columns used for merging
df.drop(['Rk', 'Squad', 'Comp', 'Player'], axis=1, inplace=True)

# %%
# Extract the 'Player_League_Club' column
player_league_club_col = df['Player_League_Club']

# Drop the 'Player_League_Club' column from its current position
df.drop('Player_League_Club', axis=1, inplace=True)

# Insert the 'Player_League_Club' column at the beginning
df.insert(0, 'Player_League_Club', player_league_club_col)

df

# %%
df.describe()

# %%
# Display the count of each unique value in the 'Pos' column
print(df['Pos'].value_counts())

# %%
## inspect FWDW, DFFW, DFMF, MFDF, MFFW, FWMF, FWMF


# Define the positions you want to inspect
positions_to_inspect = ['FWDW', 'DFFW', 'DFMF', 'MFDF', 'MFFW', 'FWMF', 'FWMF']

# Filter the DataFrame based on the specified positions
filtered_df = df[df['Pos'].isin(positions_to_inspect)]

# Display the relevant information for each player in the filtered DataFrame
filtered_df[['Pos', 'Player_League_Club']]

# %%
# Extract the corrected position from the first two letters of the 'Pos' column
df['Pos'] = df['Pos'].str[:2]

# %%
# Display unique values in the new 'Pos' column
print(df['Pos'].value_counts())


# %%
# Split the DataFrame into different datasets based on positions
df_df = df[df['Pos'] == 'DF'].copy()
df_mf = df[df['Pos'] == 'MF'].copy()
df_fw = df[df['Pos'] == 'FW'].copy()
df_gk = df[df['Pos'] == 'GK'].copy()

# %%
df_gk

# %%
html_table = df_gk.head().to_html()

# Display the HTML table
display(HTML(html_table))


# %%
# Drop columns with more than 50 '0's
df_gk = df_gk.loc[:, (df_gk == 0).sum() <= 1]

# Drop Nation, Position, Age, Born, MP, Starts, Min
gk_drop_col= ['Nation', 'Pos', 'Age', 'Born', 'MP', 'Starts', 'Min','90s']
df_gk = df_gk.drop(columns=gk_drop_col)

# %%
html_table = df_gk.head().to_html()

# Display the HTML table
display(HTML(html_table))

# %%
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Assume df_gk is your DataFrame with the features and 'Player_League_Club' column
features = df_gk.drop(['Player_League_Club'], axis=1)

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Choose the number of clusters (adjust as needed)
num_clusters = 3

# Apply K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df_gk['Cluster'] = kmeans.fit_predict(scaled_features)

def get_similar_players(player_name, num_similar_players=5):
    # Find the cluster of the given player
    player_cluster = df_gk[df_gk['Player_League_Club'] == player_name]['Cluster'].values[0]

    # Filter players in the same cluster
    similar_players = df_gk[df_gk['Cluster'] == player_cluster]

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        scaled_features[df_gk['Player_League_Club'] == player_name],
        scaled_features[similar_players['Player_League_Club']]
    )

    # Get the indices of the most similar players
    similar_indices = similarity_scores.argsort()[0][-num_similar_players-1:-1][::-1]

    # Display similar players
    st.write("Top 5 Players Similar to", player_name)
    for index in similar_indices:
        st.write(similar_players.iloc[index]['Player_League_Club'])

# Streamlit UI
st.title("Similar Goalkeepers Finder")
player_name_input = st.text_input("Enter a goalkeeper's name:")
if player_name_input:
    get_similar_players(player_name_input)


# %%



