import streamlit as st
from pydantic import BaseModel


class Boy(BaseModel):
    name: str = "john"


st.title("test app")


def write_boy():
    if "boy" in st.session_state:
        st.write(st.session_state.boy)


st.session_state.boy = Boy()
st.text_input("Your name", key="name")
st.toggle("Your name", key="on_off")
st.write("a" + st.session_state.name)
st.write(st.session_state.on_off)
st.markdown(st.session_state.boy)


def on_click_button():
    print("entered click")
    st.warning("called")
    # st.session_state.boy.name += "ab"
    st.session_state.boy = Boy(name="gabriel")
    st.session_state.name += "mais1"
    print(st.session_state.boy)
    write_boy()


st.write(st.session_state)

st.button("click me", on_click=on_click_button)
