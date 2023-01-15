import streamlit as st

st.title("Making a Button")
result = st.button("Click Here")
st.write(result)
if result:
  st.write(":smile:")

if st.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye')
