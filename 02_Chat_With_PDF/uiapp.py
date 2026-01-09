# front end for chat with pdf
import streamlit as st
from app import load_pdf, split_into_chunks, create_vectorstore, create_chat_chain

def main():
    # Streamlit app title
    st.title("Chat with PDF using LLMs")
    st.subheader("Upload a PDF and chat with it!")
    # File uploader in a sidebar
    with st.sidebar:
        st.subheader("Upload your PDF file")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
        if st.button("process") and uploaded_file is not None:
            # Load and process the PDF
            with st.spinner("Processing PDF..."):
                pdf_content = load_pdf(uploaded_file)
                chunks = split_into_chunks(pdf_content)
                vectorstore = create_vectorstore(chunks)
                st.session_state.chat_chain = create_chat_chain(vectorstore)
                st.session_state.messages = []
                st.success("PDF processed successfully! You can start chatting now.")

    # main chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_chain' not in st.session_state:
        st.info("Please upload and process a PDF file to start chatting.")
    else:
        user_input = st.text_input("You: ", key="input")
        if user_input:
            # Get response from chat chain
            chat_chain = st.session_state.chat_chain
            response = chat_chain.run(user_input)
            # Store messages in session state
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Display chat messages
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])


if __name__ == "__main__":
    main()