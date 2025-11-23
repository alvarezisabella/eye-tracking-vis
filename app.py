import pandas as pd
import plotly.express as px

df = pd.read_csv("data/raw/AOI_DGMs.csv")

fig = px.scatter(
    df,
    x="total_number_of_saccades",
    y="Approach_Score",
    color="pilot_success",
    labels={
        "total_number_of_saccades": "Total Number of Saccades",
        "Approach_Score": "Approach Score",
        "pilot_success": "Pilot Success"
    },
    title="Relationship Between Saccade Behavior and Approach Score",
    hover_data=["PID"]
)

fig.show()
