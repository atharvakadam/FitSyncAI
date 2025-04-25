import streamlit as st
from library.setup_graph import graph
from langchain_core.messages import HumanMessage

st.set_page_config(
    page_title="Health & Fitness AI Chat",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("How can I assist you with your health and fitness goals today?")
if prompt:
    # Display user message
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process the input through your AI agents here
    # For demonstration, we'll use a placeholder response
    response = f"Here's a tailored response to: {prompt}"
    # for s in graph.stream({"messages": [("user", prompt)]},subgraphs=True, stream_mode='values'):
    #     print(s[1]['messages'][-1].content)

    response = graph.invoke(input={"messages": [HumanMessage(content=prompt)]}, config={"configurable": {"thread_id": "1"}},subgraphs=True)
    
    print(response)
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response[1]["messages"][-1].content)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
