import streamlit as st
import requests
import json

st.set_page_config(page_title="Local RAG Chatbot", layout="wide")

API_URL = "http://localhost:8000/rag"

st.title("Local AI RAG Chatbot")
st.write("Ask anything")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# for msg in st.session_state["messages"]:
#     with st.chat_m

user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "question": user_input
                    }
                )
                resp_json = response.json()
                answer = resp_json.get("answer", "No answer returned")
            except Exception as e:
                answer = f"Error: {str(e)}"
        st.write(answer)
        st.session_state['messages'].append({"role": "assistant", "content": answer})