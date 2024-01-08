import streamlit as st

from enums import StyleTypes


def generate_style_section() -> tuple[str, str | None]:
    style_context = None
    col1, col2 = st.columns(2)
    with col1:
        style_type = st.radio(
            "Select the how you want to help me to style your text",
            options=[style for style in StyleTypes],
            index=0,
        )
    with col2:
        if style_type == StyleTypes.FREE:
            st.write("We will leave my AI brain to do the magic")
        elif style_type == StyleTypes.CONCRETE:
            style_context = st.text_input(
                "Please type your desired style in one word or short sentence",
                placeholder="e.g. formal, informal, friendly, etc.",
            )
        elif style_type == StyleTypes.RULES:
            st.caption("I.e. Use a lot of icons!!!")
            def add_rule():
                st.session_state.style_rules_size += 1

            def delete_rule(index):
                st.session_state.style_rules_size -= 1
                del st.session_state.style_rules[index]
                del st.session_state.style_rules_del[index]

            if "style_rules_size" not in st.session_state:
                st.session_state.style_rules_size = 0
                st.session_state.style_rules = []
                st.session_state.style_rules_del = []

            # fields and types of the table
            for i in range(st.session_state.style_rules_size):
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state.style_rules.append(st.text_input(f"Rule {i}", key=f"text{i}"))

                with c2:
                    st.session_state.style_rules_del.append(st.button("❌", key=f"delete{i}", on_click=delete_rule, args=(i,)))

            st.button("➕ Add Rule", on_click=add_rule)
        elif style_type == StyleTypes.WEBPAGE:
            url = st.text_input("Please type the url of the webpage you want to use as a reference")
            st.session_state.webpage = url
        elif style_type == StyleTypes.DOCUMENT:
            doc = st.file_uploader("Upload your style document", type=["pdf", "docx", "txt", "odt"])
            st.session_state.document = doc
        return style_type, style_context