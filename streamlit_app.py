import streamlit as st
from datetime import time

# st.title("Making a Button")
# result = st.button("Click Here")
# st.write(result)
# if result:
#   st.write(":smile:")

# if st.button('Say hello'):
#     st.write('Why hello there')
# else:
#     st.write('Goodbye')

# age = st.slider('How old are you?', 0, 130, 25)
# st.write("I'm ", age, 'years old')

# values = st.slider(
#     'Select a range of values',
#     0.0, 100.0, (25.0, 75.0))
# st.write('Values:', values)

# appointment = st.slider(
#     "Schedule your appointment:",
#     value=(time(11, 30), time(12, 45)))
# st.write("You're scheduled for:", appointment)
    
# start_time = st.slider(
#     "When do you start?",
#     value=datetime(2020, 1, 1, 9, 30),
#     format="MM/DD/YY - hh:mm")
# st.write("Start time:", start_time)


page_names = ['Checkbox', 'Button']
page = st.radio('Navigation', page_names)
st.write("**The variable 'page' returns:**", page)

if page == 'Checkbox':
   st.subheader('Welcome to the Checkbox page!')
   st.write("Nice to see you! :wave:")

   check = st.checkbox("Click here")
   st.write('State of the checkbox:', check)

   if check:
       nested_btn = st.button("Button nested in Checkbox")

       if nested_btn:
           st.write(":cake:"*20)
else:
   st.subheader("Welcome to the Button page!")
   st.write(":thumbsup:")

   result = st.button('Click Here')
   st.write("State of button:",result)

   if result:
       nested_check = st.checkbox("Checkbox nested in Button")

       if nested_check:
           st.write(":heart:"*20)
