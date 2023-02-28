import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

st.title('PVP Balance Sheets')


@st.cache(allow_output_mutation=True)
def load_dataset():
    import requests
    url = 'https://api.gotchiverse.io/leaderboard/all'
    results = []
    for offset in range(0, 4001, 20):
        response = requests.get(url, params={'limit': 20, 'offset': offset})
        results.extend(response.json()['leaderboard'])
    for result in results:
        for i, k in result['damageDealtByType'].items():
            result['damageDealtByType.' + i] = k
        for i, k in result['damageTakenByType'].items():
            result['damageTakenByType.' + i] = k
        for i, k in result['deathsByType'].items():
            result['deathsByType.' + i] = k
        for i, k in result['killsByType'].items():
            result['killsByType.' + i] = k

    dataframe = pd.DataFrame(results)
    dataframe = dataframe.drop(['tips','destructibles','damageDealtByType','damageTakenByType','deathsByType','killsByType'],axis = 1)
    dataframe['sessionTimeMins'] = dataframe['sessionTime'] / 60
    dataframe['SessionTimeHours'] = dataframe['sessionTimeMins'] / 60
    ls_ids = list(dataframe['id'].dropna().astype(int).values)
    query = f'''
    {{
     aavegotchis(orderBy:gotchiId,first:1000,where:{{gotchiId_in:{ls_ids}}}) {{
      gotchiId
      modifiedNumericTraits
      collateral
      hauntId
      level
      name
      owner {{
        id
      }}
     }}
    }}
    '''

    def rorm(x):
        if x > 50:
            return 'Ranged'
        else:
            return 'Melee'

    def run_query(query):

        request = requests.post('https://subgraph.satsuma-prod.com/tWYl5n5y04oz/aavegotchi/aavegotchi-core-matic/api'
                                '',
                                json={'query': query})
        if request.status_code == 200:
            return request.json()
        else:
            raise Exception('Query failed. return code is {}.      {}'.format(request.status_code, query))

    traits_data = run_query(query)
    for gotchi in traits_data['data']['aavegotchis']:
        gotchi['NRG'] = gotchi['modifiedNumericTraits'][0]
        gotchi['AGG'] = gotchi['modifiedNumericTraits'][1]
        gotchi['SPK'] = gotchi['modifiedNumericTraits'][2]
        gotchi['BRN'] = gotchi['modifiedNumericTraits'][3]
    df_traits = pd.DataFrame(traits_data['data']['aavegotchis'])
    df_traits['RorM'] = df_traits['BRN'].apply(lambda x: (rorm((x))))
    ls_sprinter_rangeds = np.where(
        ((df_traits.NRG > 50) & ((df_traits.AGG > 50)) & ((df_traits.SPK <= 50)) & ((df_traits.BRN > 50))))
    ls_healthy_sprinter_rangeds = np.where(
        ((df_traits.NRG <= 50) & ((df_traits.AGG > 50)) & ((df_traits.SPK <= 50)) & ((df_traits.BRN > 50))))
    ls_ethereal_rangers = np.where(
        ((df_traits.NRG > 50) & ((df_traits.AGG > 50)) & ((df_traits.SPK > 50)) & ((df_traits.BRN > 50))))
    ls_stone_rangers = np.where(
        ((df_traits.NRG > 50) & ((df_traits.AGG <= 50)) & ((df_traits.SPK <= 50)) & ((df_traits.BRN > 50))))
    ls_diamond_rangers = np.where(
        ((df_traits.NRG <= 50) & ((df_traits.AGG <= 50)) & ((df_traits.SPK <= 50)) & ((df_traits.BRN > 50))))
    ls_explosive_rangers = np.where(
        ((df_traits.NRG <= 50) & ((df_traits.AGG > 50)) & ((df_traits.SPK > 50)) & ((df_traits.BRN > 50))))
    ls_ethereal_stone_rangers = np.where(
        ((df_traits.NRG > 50) & ((df_traits.AGG <= 50)) & ((df_traits.SPK > 50)) & ((df_traits.BRN > 50))))
    ls_ethereal_diamong_rangers = np.where(
        ((df_traits.NRG <= 50) & ((df_traits.AGG <= 50)) & ((df_traits.SPK > 50)) & ((df_traits.BRN > 50))))

    ls_sprinter_warriors = np.where(
        ((df_traits.NRG > 50) & ((df_traits.AGG > 50)) & ((df_traits.SPK <= 50)) & ((df_traits.BRN <= 50))))
    ls_healthy_sprinter_warriors = np.where(
        ((df_traits.NRG <= 50) & ((df_traits.AGG > 50)) & ((df_traits.SPK <= 50)) & ((df_traits.BRN <= 50))))
    ls_ethereal_warriors = np.where(
        ((df_traits.NRG > 50) & ((df_traits.AGG > 50)) & ((df_traits.SPK > 50)) & ((df_traits.BRN <= 50))))
    ls_stone_warriors = np.where(
        ((df_traits.NRG > 50) & ((df_traits.AGG <= 50)) & ((df_traits.SPK <= 50)) & ((df_traits.BRN <= 50))))
    ls_diamond_warriors = np.where(
        ((df_traits.NRG <= 50) & ((df_traits.AGG <= 50)) & ((df_traits.SPK <= 50)) & ((df_traits.BRN <= 50))))
    ls_explosive_warriors = np.where(
        ((df_traits.NRG <= 50) & ((df_traits.AGG > 50)) & ((df_traits.SPK > 50)) & ((df_traits.BRN <= 50))))
    ls_ethereal_stone_warriors = np.where(
        ((df_traits.NRG > 50) & ((df_traits.AGG <= 50)) & ((df_traits.SPK > 50)) & ((df_traits.BRN <= 50))))
    ls_ethereal_diamong_warriors = np.where(
        ((df_traits.NRG <= 50) & ((df_traits.AGG <= 50)) & ((df_traits.SPK > 50)) & ((df_traits.BRN <= 50))))
    for i in ls_sprinter_rangeds:
        df_traits.loc[i, 'Types'] = 'Ranged Sprinter'
    for i in ls_healthy_sprinter_rangeds:
        df_traits.loc[i, 'Types'] = 'Healthy Ranged Sprinter'
    for i in ls_ethereal_rangers:
        df_traits.loc[i, 'Types'] = 'Ethereal Ranger'
    for i in ls_stone_rangers:
        df_traits.loc[i, 'Types'] = 'Stone Ranger'
    for i in ls_diamond_rangers:
        df_traits.loc[i, 'Types'] = 'Diamond Ranger'
    for i in ls_explosive_rangers:
        df_traits.loc[i, 'Types'] = 'Explosive Ranger'
    for i in ls_ethereal_stone_rangers:
        df_traits.loc[i, 'Types'] = 'Ethereal Stone Ranger'
    for i in ls_ethereal_diamong_rangers:
        df_traits.loc[i, 'Types'] = 'Ethereal Diamond Ranger'

    for i in ls_sprinter_warriors:
        df_traits.loc[i, 'Types'] = 'Sprinter Warrior'
    for i in ls_healthy_sprinter_warriors:
        df_traits.loc[i, 'Types'] = 'Healthy Sprinter Warrior'
    for i in ls_ethereal_warriors:
        df_traits.loc[i, 'Types'] = 'Ethereal Warrior'
    for i in ls_stone_warriors:
        df_traits.loc[i, 'Types'] = 'Stone Warrior'
    for i in ls_diamond_warriors:
        df_traits.loc[i, 'Types'] = 'Diamond Warrior'
    for i in ls_explosive_warriors:
        df_traits.loc[i, 'Types'] = 'Explosive Warrior'
    for i in ls_ethereal_stone_warriors:
        df_traits.loc[i, 'Types'] = 'Ethereal Stone Warrior'
    for i in ls_ethereal_diamong_warriors:
        df_traits.loc[i, 'Types'] = 'Ethereal Diamond Warrior'

    df_traits['gotchiId'] = df_traits['gotchiId'].astype(int)
    dataframe = dataframe.dropna()
    dataframe['id'] = dataframe['id'].astype(int)
    df_merged = dataframe.merge(df_traits, how='left', left_on='id', right_on='gotchiId').dropna()
    df_merged['KDR'] = df_merged['kills'] / df_merged['deaths']
    df_merged = df_merged[df_merged['deaths'] > 0]
    df_merged = df_merged[df_merged['kills'] > 0]
    return df_merged


import pandas as pd

df = load_dataset()
st.dataframe(df)
df['KDR'] = df['kills'] / df['deaths']
df = df[(df['kills'] > 0) & (df['deaths'] > 0)]
df_kdr = df.groupby(['Types'])['KDR'].mean().sort_values(ascending=False)
df['sessionTimeMins'] = df['sessionTime'] / 60
df['SessionTimeHours'] = df['sessionTimeMins'] / 60
df['KDR'] = df['kills'] / df['deaths']
kpi_mins = df['sessionTimeMins'].mean()
kpi_hours = df['SessionTimeHours'].sum()
kpi_ranged = len(df[df['RorM'] == 'Ranged'])
kpi_melee = len(df[df['RorM'] == 'Melee'])
kpi_kdr_melee = df[df['RorM'] == 'Melee']['KDR'].mean()
kpi_kdr_ranged = df[df['RorM'] == 'Ranged']['KDR'].mean()

col1, col2 = st.columns(2)
with col1:
    st.metric('Average minutes played', round(kpi_mins, 2),delta = round(kpi_mins - 31 ,2))
    st.metric('Number of Melee gotchis', kpi_melee,delta = kpi_melee - 129)
    st.metric('Number of Ranged gotchis', kpi_ranged,delta = round(kpi_ranged - 165 ,2))

with col2:
    st.metric('Total hours played', round(kpi_hours, 2),delta = round(kpi_hours - 152,2))
    st.metric('KDR Melee mean', round(kpi_kdr_melee, 2), delta = round(kpi_kdr_melee - 1.24 ,2))
    st.metric('KDR Ranged mean', round(kpi_kdr_ranged, 2), delta = round(kpi_kdr_ranged -0.92,2))

    df_filtered = df.groupby(['Types', 'RorM'])[['RorM', 'KDR']].mean().sort_values(
        by='KDR', ascending=False)
    df_reindex = []
    for t1, t2 in df_filtered.index:
        df_reindex.append(str(t1))


fig5 = make_subplots(rows=3, cols=3, subplot_titles=('Rush vs Snipe DMG', 'Melee vs Range DMG', 'Melee vs Range Kills','KDR mean by Type','Types Count','Minutes played total per Type'))
fig5.add_trace(go.Bar(x=["RushDMG", "SnipeDMG"],
                      y=[df['damageDealtByType.rush'].sum(), df['damageDealtByType.snipe'].sum()]), 1, 1)
fig5.add_trace(go.Bar(x=["MeleeDMG", "RangedDMG"],
                      y=[df['damageDealtByType.melee'].sum(), df['damageDealtByType.range'].sum()]), 1, 2)
fig5.add_trace(
    go.Bar(x=["MeleeKills", "RangedKills"], y=[df['killsByType.melee'].sum(), df['killsByType.range'].sum()]), 1, 3)
fig5.add_trace(
    go.Bar(x=df_reindex,
                  y=df_filtered['KDR'].values), 2, 1)
fig5.add_trace(go.Bar(x = df['Types'].value_counts().index,
                     y = df['Types'].value_counts().values), 2, 2)
fig5.add_trace(go.Bar(x = df.groupby(['Types'])['sessionTimeMins'].sum().sort_values(ascending=False).index,
                     y = df.groupby(['Types'])['sessionTimeMins'].sum().sort_values(ascending=False).values), 2, 3)
fig5.update_layout(showlegend=False, height=900, width=900)


st.plotly_chart(fig5)
