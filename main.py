import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
st.set_page_config(page_title="Aavegotchi", page_icon="chart_with_upwards_trend", layout='centered', initial_sidebar_state='auto')
st.title('PVP Balance Sheets')
st.image('aarena.png')
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
        for i, k in result['tips'].items():
            result['tip.' + i] = k

    dataframe = pd.DataFrame(results)
    dataframe = dataframe.drop(['destructibles','damageDealtByType','damageTakenByType','deathsByType','killsByType'],axis = 1)
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
    df_merged['owner'] = df_merged['owner'].apply(lambda x: x['id'])
    return df_merged


import pandas as pd

df = load_dataset()

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
kpi_addresses = df['owner'].nunique()
kpi_FUD = df['tip.FUD'].sum()
kpi_FOMO = df['tip.FOMO'].sum()
kpi_ALPHA = df['tip.ALPHA'].sum()
kpi_KEK = df['tip.KEK'].sum()
col1, col2, col3,col4 = st.columns(4)
with col1:
    st.metric('Average minutes played', round(kpi_mins, 2),delta = round(kpi_mins - 31 ,2))
    st.metric('Number of Melee gotchis', kpi_melee,delta = kpi_melee - 129)
    st.metric('Number of Ranged gotchis', kpi_ranged,delta = round(kpi_ranged - 165 ,2))

with col2:
    st.metric('Total hours played', round(kpi_hours, 2),delta = round(kpi_hours - 152,2))
    st.metric('KDR Melee mean', round(kpi_kdr_melee, 2), delta = round(kpi_kdr_melee - 1.24 ,2),help = 'Kill Death Ratio')
    st.metric('KDR Ranged mean', round(kpi_kdr_ranged, 2), delta = round(kpi_kdr_ranged -0.92,2),help = 'Kill Death Ratio')

    df_filtered = df.groupby(['Types', 'RorM'])[['RorM', 'KDR']].mean().sort_values(
        by='KDR', ascending=False)
    df_reindex = []
    for t1, t2 in df_filtered.index:
        df_reindex.append(str(t1))
with col3:
    st.metric('Unique addresses',kpi_addresses, delta = kpi_addresses - 196)
    st.metric('FUD tipped',round(kpi_FUD),delta = round(kpi_FUD))
    st.metric('FOMO tipped',round(kpi_FOMO),delta = round(kpi_FOMO))

with col4:
    st.metric('Alpha tipped', round(kpi_ALPHA),delta = round(kpi_ALPHA))
    st.metric('KEK tipped', round(kpi_KEK), delta = round(kpi_KEK))
    st.metric('Highest mean KDR',df.groupby(['Types'])['KDR'].mean().sort_values(ascending = False).index[0],help = df.groupby(['Types'])['KDR'].mean().sort_values(ascending = False).index[0])


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
fig5.add_trace(go.Bar(x=["RushKILLS", "SnipeKILLS"],
                      y=[df['killsByType.rush'].sum(), df['killsByType.snipe'].sum()]), 3, 1)

fig5.update_layout(showlegend=False, height=900, width=900)


st.plotly_chart(fig5)

st.subheader('Different Gotchi Classes')
col_mk,col_mk2 = st.columns(2)
with col_mk:
    st.markdown('''
    NRG + AGG + SPK - BRN + ---------- Sprinter Ranger  \t  **Very effective sprinting combo dps chase you** \n
    NRG - AGG + SPK - BRN + ---------- Healthy Sprinter Ranger\t **More hp at the cost of less total AP**\n
    NRG + AGG + SPK + BRN + ---------- Ethereal Ranger\t **No AP regen but gains evasion** \n
    NRG + AGG - SPK - BRN + ---------- Stone Ranger\t **Range tank that can hit hard and has quite AP**\n
    NRG - AGG - SPK - BRN + ---------- Diamond Ranger\t **Range tank with less ap but Healthy**\n
    NRG - AGG + SPK + BRN + ---------- Explosive Ranger\t **Low AP low AP Regen they nuke and empty fast**\n
    NRG + AGG - SPK + BRN + ---------- Ethereal Stone Ranger\t **Tanky and evasive range dps?**\n 
    NRG - AGG - SPK + BRN + ---------- Ethereal Diamond Ranger\t **Tanky evasive and healthy but no AP at all**\n''')
with col_mk2:
    st.markdown('''
NRG + AGG + SPK - BRN - ---------- Sprinter Warrior\t **Counter melee part** \n
NRG - AGG + SPK - BRN - ---------- Healthy Sprinter Warrior\t **Counter melee part** \n
NRG + AGG + SPK + BRN - ---------- Ethereal Warrior\t **Counter melee part** \n
NRG + AGG - SPK - BRN - ---------- Stone Warrior\t **Counter melee part** \n
NRG - AGG - SPK - BRN - ---------- Diamond Warrior\t **Counter melee part** \n
NRG - AGG + SPK + BRN - ---------- Explosive Warrior\t **Counter melee part** \n
NRG + AGG - SPK + BRN - ---------- Ethereal Stone Warrior\t **Counter melee part** \n
NRG - AGG - SPK + BRN - ---------- Ethereal Diamond Warrior\t **Counter melee part** \n
''')

st.title('Traits Distribution')

import plotly.express as px


fig_box = px.box(df, y="NRG", color="RorM",
             notched=True, # used notched shape
             title="Box plot NRG",
             hover_data=["RorM"] # add day column to hover data
            )
fig_box.update_layout(height=350, width=350)
fig_box2 = px.box(df, y="AGG", color="RorM",
             notched=True, # used notched shape
             title="Box plot AGG",
             hover_data=["RorM"] # add day column to hover data
            )
fig_box2.update_layout(height=350, width=350)
cols1,cols2 = st.columns(2)
with cols1:
    st.plotly_chart(fig_box,height=300,width=300)
with cols2:
    st.plotly_chart(fig_box2,heigh=100,width=100)
cols3,cols4 = st.columns(2)
fig_box3 = px.box(df, y="SPK", color="RorM",
             notched=True, # used notched shape
             title="Box plot SPK",
             hover_data=["RorM"] # add day column to hover data
            )
fig_box3.update_layout(height=350, width=350)
fig_box4 = px.box(df, y="BRN", color="RorM",
             notched=True, # used notched shape
             title="Box plot BRN",
             hover_data=["RorM"] # add day column to hover data
            )
fig_box4.update_layout(height=350, width=350)
with cols3:
    st.plotly_chart(fig_box3)
with cols4:
    st.plotly_chart(fig_box4)


df['HPK'] = df['hits']/df['kills']
df['HPM'] = df['hits']/df['sessionTimeMins']


labels1 = df.groupby(['Types'])['HPM'].mean().sort_values(ascending = False).index
values1 = df.groupby(['Types'])['HPM'].mean().sort_values(ascending = False).values
labels2 = df.groupby(['Types'])['deaths'].mean().sort_values(ascending = False).index
values2 = df.groupby(['Types'])['deaths'].mean().sort_values(ascending = False).values
labels3 = df.groupby(['Types'])['kills'].mean().sort_values(ascending = False).index
values3 = df.groupby(['Types'])['kills'].mean().sort_values(ascending = False).values
labels4 = df.groupby(['Types'])['hits'].mean().sort_values(ascending = False).index
values4 = df.groupby(['Types'])['hits'].mean().sort_values(ascending = False).values
labels5 = df.groupby(['Types'])['sessionTimeMins'].mean().sort_values(ascending = False).index
values5 = df.groupby(['Types'])['sessionTimeMins'].mean().sort_values(ascending = False).values
labels6 = df.groupby(['Types'])['HPK'].mean().sort_values(ascending = False).index
values6 = df.groupby(['Types'])['HPK'].mean().sort_values(ascending = False).values
st.title('Types Analysis')
# Use `hole` to create a donut-like pie chart
pie1,pie2 = st.columns(2)
with pie1:
    fig_pie = go.Figure(data=[go.Pie(labels=labels1, values=values1, hole=.3)])
    fig_pie.update_layout(showlegend=True, height=400, width=400, title='Hits / Minute Avg')
    st.plotly_chart(fig_pie, height=400, width=400)
with pie2:
    fig_pie2 = go.Figure(data=[go.Pie(labels=labels2, values=values2, hole=.3)])
    fig_pie2.update_layout(showlegend=True, height=400, width=400, title='Deaths Avg')
    st.plotly_chart(fig_pie2, height=400, width=400)

pie3,pie4 = st.columns(2)
with pie3:
    fig_pie = go.Figure(data=[go.Pie(labels=labels3, values=values3, hole=.3)])
    fig_pie.update_layout(showlegend=True, height=400, width=400, title='Kills Avg')
    st.plotly_chart(fig_pie, height=400, width=400)
with pie4:
    fig_pie2 = go.Figure(data=[go.Pie(labels=labels4, values=values4, hole=.3)])
    fig_pie2.update_layout(showlegend=True, height=400, width=400, title='Avg Hits')
    st.plotly_chart(fig_pie2, height=400, width=400)
pie5,pie6 = st.columns(2)
with pie5:
    fig_pie = go.Figure(data=[go.Pie(labels=labels5, values=values5, hole=.3)])
    fig_pie.update_layout(showlegend=True, height=400, width=400, title='Avg Mins Played')
    st.plotly_chart(fig_pie, height=400, width=400)
with pie6:
    fig_pie2 = go.Figure(data=[go.Pie(labels=labels6, values=values6, hole=.3)])
    fig_pie2.update_layout(showlegend=True, height=400, width=400, title='Hits/Kills avg')
    st.plotly_chart(fig_pie2, height=400, width=400)


import plotly.graph_objects as go

df = pd.DataFrame(dict(
    value=[99, 101, 90, 95,55,70,65,80],
    variable=['BRN', 'AGG', 'SPK', 'NRG', 'BRN', 'AGG', 'SPK', 'NRG'],
    gotchis=['Gotchi1', 'Gotchi1', 'Gotchi1', 'Gotchi1',
               'Gotchi2', 'Gotchi2', 'Gotchi2', 'Gotchi2']))

fig = px.line_polar(df, r='value', theta='variable', line_close=True,
                    color='gotchis')
fig.update_traces(fill='toself')

# st.pyplot(fig3)

st.markdown("<h1 style='text-align: center; color: blue;'>CRS gotchi comparator in the works</h1>",
            unsafe_allow_html=True)
st.plotly_chart(figure_or_data=fig)
