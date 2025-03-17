from openai import OpenAI
import streamlit as st

st.title("ChatGPT-like clone")

client = OpenAI(
    api_key='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'
)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
