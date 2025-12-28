# run: streamlit run 11_streamlit_frontend.py

from streamlit.streamlit_backend import chatbot
import streamlit as st
from langchain_core.messages import HumanMessage

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


# === Loading conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


user_input = st.chat_input('Type here...')

if user_input:

# === Current user message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

# === Current assistant message
    config = {'configurable': {'thread_id': 'thread_id_1'}}
    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=config)
    ai_message = response['messages'][-1].content

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)