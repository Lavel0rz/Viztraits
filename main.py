import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.title('PVP Balance Sheets')
text = st.text_input('Enter google sheet URL')
text2 = st.text_input('Enter sheet name')
import pandas as pd
try:
    url = f"https://docs.google.com/spreadsheets/d/{text}/gviz/tq?tqx=out:csv&sheet={text2}"
    df = pd.read_csv(url)
    df = df[(df['kills'] > 0) & (df['deaths'] > 0)]
    df_kdr = df.groupby(['Type'])['killDeathRatio'].mean().sort_values(ascending = False)
    df['sessionTimeMins']= df['sessionTime']/60
    df['SessionTimeHours'] = df['sessionTimeMins']/60
    df['KDR'] = df['kills'] / df['deaths']
    kpi_mins = df['sessionTimeMins'].mean()
    kpi_hours = df['SessionTimeHours'].sum()
    kpi_ranged = len(df[df['Traits: Ranged/Melee']=='Ranged'])
    kpi_melee = len(df[df['Traits: Ranged/Melee']=='Melee'])
    kpi_kdr_melee = df[df['Traits: Ranged/Melee']=='Melee']['KDR'].mean()
    kpi_kdr_ranged = df[df['Traits: Ranged/Melee'] == 'Ranged']['KDR'].mean()
    st.dataframe(df)
    col1, col2 = st.columns(2)
    with col1:
        st.metric('Average minutes played',round(kpi_mins,2))
        st.metric('Number of Melee gotchis',kpi_melee )
        st.metric('Number of Ranged gotchis', kpi_ranged)

    with col2:
        st.metric('Total hours played',round(kpi_hours,2))
        st.metric('KDR Melee mean',round(kpi_kdr_melee,2))
        st.metric('KDR Ranged mean', round(kpi_kdr_ranged,2))

    fig = px.bar(df, x=["MeleeKills", "RangedKills"], y=[df['killsByType.mele'].sum(), df['killsByType.range'].sum()],text_auto=True,title= 'Ranged vs Melee Total Kills')
    fig.update_layout(
        title=go.layout.Title(
            text="Ranged vs Melee Total Kills",
            xref="paper",
            x=0
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text="Melee / Range",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="#7f7f7f"
                )
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text="Total Kills",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="#7f7f7f"
                )
            )
        )
    )

    col3,col4,col5,col6,col7,col8 = st.columns(6)
    fig2 = px.bar(df, x=["RushDMG", "SnipeDMG"],
                  y=[df['damageDealtByType.rush'].sum(), df['damageDealtByType.snipe'].sum()],
                  text_auto=True)
    fig2.update_layout(
        title=go.layout.Title(
            text="Rush vs Snipe Total Damage",
            xref="paper",
            x=0
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text="Rush / Snipe",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="#7f7f7f"
                )
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text="Total Damage",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="#7f7f7f"
                )
            )
        )
    )

    with col3:
        st.plotly_chart(fig)
        st.plotly_chart(fig2)
        df_filtered = df.groupby(['Type', 'Traits: Ranged/Melee'])[['Traits: Ranged/Melee', 'KDR']].mean().sort_values(
            by='KDR', ascending=False)
        df_reindex = []
        for t1, t2 in df_filtered.index:
            df_reindex.append(str(t1) + ' ' + t2)
        fig3 = px.bar(df, x=df_reindex,
                      y=df_filtered['KDR'].values,
                      text_auto=True)
        fig3.update_layout(
            title=go.layout.Title(
                text="KDR by Type and Melee/Range",
                xref="paper",
                x=0
            ),
            xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                    text="Types",
                    font=dict(
                        family="Courier New, monospace",
                        size=18,
                        color="#7f7f7f"
                    )
                )
            ),
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                    text="KDR",
                    font=dict(
                        family="Courier New, monospace",
                        size=18,
                        color="#7f7f7f"
                    )
                )
            )
        )
        st.plotly_chart(fig3)



except:
    st.warning('Please fill the inputs')
