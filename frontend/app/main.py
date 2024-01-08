import os

import streamlit as st
import requests

from enums import OpenAIModels, TeacherActions
from sections.style import generate_style_section
from sections.summarization import generate_summarization_section


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/api/stream")


def send_request(text, model, action, summarization_type, style_type, style_context):
    result_container = st.empty()
    buffer = ""
    res = requests.post(
        BACKEND_URL,
        params={
            "text": text,
            "model": model.value,
            "action": action.value,
            "summarization_type": summarization_type,
            "style_type": style_type,
            "style_context": style_context,
            "style_rules": [""],
            "webpage": st.session_state.get("webpage"),
        },
        files={"file": st.session_state.get("document")},
        stream=True,
    )
    result_container.markdown("Here you have my corrections:")
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        for chunk in res.iter_content(None, decode_unicode=True):

            if chunk:
                buffer += str(chunk)
                result_container.markdown(f"{buffer}")


style_type, style_context, summarization_type = None, None, None

st.write("# Welcome to favourite AI english classroom! 👋")
st.title("# AI 🤖 English Teacher 🧑‍🏫")
st.subheader("Powered by Streamlit + FastAPI + LangChain + OpenAI")

model = st.selectbox(
    "Select your favourite AI Teacher",
    options=[model for model in OpenAIModels],
    index=len(OpenAIModels) - 1
)

action = st.selectbox(
    "What do you want from your AI Teacher?",
    options=[action for action in TeacherActions],

)

if action == TeacherActions.STYLE:
    st.markdown("""---""")
    style_type, style_context = generate_style_section()

if action == TeacherActions.SUMMARIZATION:
    st.markdown("""---""")
    summarization_type = generate_summarization_section()

st.markdown("""---""")

if action == TeacherActions.SUMMARIZATION and st.session_state.get("webpage") or st.session_state.get("document"):
    if st.button("Prompt"):
        send_request(
            text="summary",
            model=model,
            action=action,
            summarization_type=summarization_type,
            style_type=style_type,
            style_context=style_context,
        )

else:
    text = st.text_area(
        f"Please write the text to perform {action.value.lower()}", value="This are a rally bad text wrote in anglish"
    )
    if st.button("Prompt"):
        send_request(
            text=text,
            model=model,
            action=action,
            summarization_type=summarization_type,
            style_type=style_type,
            style_context=style_context,
        )
