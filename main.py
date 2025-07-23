import streamlit as st
import random
import time
from generate import simplified, capitalize_ent

if 'text_simplified' not in st.session_state:
   st.session_state.text_simplified = simplified()

st.title("🤖CyBERT - TS")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# Streamed response emulator
def response_generator():
    # response = random.choice(
    #     [
    #         "Hello there! How can I assist you today?",
    #         "Hi, human! Is there anything I can help you with?",
    #         "Do you need help?",
    #     ]
    # )
    if prompt == None:
        response = random.choice(["Feel free to send me sentences to simplify!", "I am ready to assist with text simplification!"])
        for word in response.split():
            yield word + " "
            time.sleep(0.05)
    else:
        seq_len = st.session_state.text_simplified.load_data(prompt)
        response = st.session_state.text_simplified.generate()
        temp_resp = ''
        for _ in range(seq_len):
            try:
                resp = next(response)
            except:
                break
            if seq_len == 0 and len(resp) > 1:
                temp_resp = resp[0].upper() + resp[1]

            else:
                if '#' in resp:
                    # print("# in resp")
                    temp_resp += resp.replace('#', '')
                else:
                    temp_resp = capitalize_ent(temp_resp)
                    yield  temp_resp + " "
                    temp_resp = resp
                time.sleep(0.05)


            # for word in temp_resp.split():
            # if '#' in resp:
            #     temp_resp += resp
            #     # yield temp_resp.replace('#', '')
            # else:
            #     temp_resp = capitalize_ent(temp_resp)
            #     if seq_len == 0 and len(resp) > 1:
            #         resp = resp[0].upper() + resp[1:]
            #     yield  resp + " "
            #     temp_resp = ''
            # time.sleep(0.05)

# Display assistant response in chat message container
with st.chat_message("assistant"):
    response = st.write_stream(response_generator())
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})