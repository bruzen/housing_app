import altair as alt
import pandas as pd
import streamlit as st
from vega_datasets import data

@st.experimental_memo
def get_data():
    source = data.stocks()
    source = source[source.date.gt("2004-01-01")]
    return source

source = get_data()

# Original time series chart. Omitted `get_chart` for clarity
chart = get_chart(source)

# Input annotations
ANNOTATIONS = [
    ("Mar 01, 2008", "Pretty good day for GOOG"),
    ("Dec 01, 2007", "Something's going wrong for GOOG & AAPL"),
    ("Nov 01, 2008", "Market starts again thanks to..."),
    ("Dec 01, 2009", "Small crash for GOOG after..."),
]

# Create a chart with annotations
annotations_df = pd.DataFrame(ANNOTATIONS, columns=["date", "event"])
annotations_df.date = pd.to_datetime(annotations_df.date)
annotations_df["y"] = 0
annotation_layer = (
    alt.Chart(annotations_df)
    .mark_text(size=15, text="⬇", dx=-14, dy=-24, align="center")
    .encode(
        x="date:T",
        y=alt.Y("y:Q"),
        tooltip=["event"],
    )
    .interactive()
)

# Display both charts together
st.altair_chart((chart + annotation_layer).interactive(), use_container_width=True)
