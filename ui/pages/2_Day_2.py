import streamlit as st
from utils import post_json

st.set_page_config(page_title="Day 2 - Reasoning Basics", page_icon="🧠")

st.title("Day 2 · Self-Consistency / Tool-Use / Plan-and-Solve (single input)")

option = st.selectbox(
    "Which core task version do you want to use?",
    (
        "core_task_2_bonus",
        "core_task_2",
        "core_task_1",
    ),
)

prompt = st.chat_input("Enter a question or task")
if prompt:
    with st.spinner("Calling backend..."):
        data = post_json(f"/api/day2/{option}", {"message": prompt})
    st.write(data.get("reply", "<no reply>"))
