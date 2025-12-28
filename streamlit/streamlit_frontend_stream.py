# run: streamlit run streamlit_frontend_stream.py

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
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config={'configurable': {'thread_id': 'thread_1'}},
                stream_mode='messages'
            )
        )
    
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
        