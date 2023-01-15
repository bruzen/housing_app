import streamlit as st

st.title("Making a Button")
result = st.button("Click Here")
st.write(result)
if result:
  st.write(":smile:")
