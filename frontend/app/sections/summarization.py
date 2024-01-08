import streamlit as st

from enums import SummarizationTypes


def generate_summarization_section() -> str:
    col1, col2 = st.columns(2)

    with col1:
        summarization_type = st.radio(
            "Select the how you want to help me to summarize your text",
            options=[summarization for summarization in SummarizationTypes],
            index=0,
        )
    with col2:
        if summarization_type == SummarizationTypes.WEBPAGE:
            url = st.text_input("Please type the url of the webpage you want to make a summary of")
            st.session_state.webpage = url
        elif summarization_type == SummarizationTypes.DOCUMENT:
            doc = st.file_uploader("Upload your document", type=["pdf", "txt"])
            st.session_state.document = doc

    return summarization_type
