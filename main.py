import streamlit as st
from PIL import Image
import google.generativeai as genai
from gemini_helper import GeminiEstimator

# Page configuration
st.set_page_config(
    page_title="WSEC Assistant",
    page_icon="🏗️",
    layout="wide"
)

# Initialize session state
if 'chat' not in st.session_state:
    st.session_state.chat = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'images_analyzed' not in st.session_state:
    st.session_state.images_analyzed = False
if 'current_images' not in st.session_state:
    st.session_state.current_images = []

# Sidebar for API key
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    if api_key:
        estimator = GeminiEstimator(api_key)
    else:
        estimator = GeminiEstimator()
    
    # Add clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat = estimator.start_chat()
        st.session_state.images_analyzed = False
        st.session_state.current_images = []
        st.rerun()

# Main title
st.title("🏗️ WSEC Assistant")
st.markdown("Chat about the Washington State Energy Code or upload construction plan images for analysis")

# Initialize chat if not already done
if st.session_state.chat is None:
    st.session_state.chat = estimator.start_chat()

# File uploader
uploaded_files = st.file_uploader("Upload construction plan images (optional)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

# Display uploaded images if any
if uploaded_files:
    # Create columns for image display
    cols = st.columns(3)
    
    # Display images in a grid
    for idx, uploaded_file in enumerate(uploaded_files):
        col_idx = idx % 3
        with cols[col_idx]:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Plan Image {idx + 1}", use_container_width=True)
    
    # Analyze button
    if not st.session_state.images_analyzed and len(uploaded_files) > 0:
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Generate Labor Estimate", type="primary"):
                with st.spinner("Analyzing plans and calculating labor estimates..."):
                    # Store images for reference
                    st.session_state.current_images = [Image.open(file) for file in uploaded_files]
                    
                    # Get initial analysis
                    report = estimator.analyze_images(st.session_state.current_images, st.session_state.chat)
                    
                    # Add the report to chat history
                    st.session_state.messages.append({"role": "assistant", "content": report})
                    st.session_state.images_analyzed = True
                    st.rerun()

# Display chat history
st.markdown("### 💬 Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input - Now always available
if prompt := st.chat_input("Ask questions about the Washington State Energy Code..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = estimator.send_message(st.session_state.chat, prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Footer with instructions
st.markdown("---")
st.markdown("""
### How to Use This Assistant
1. Start chatting right away about the Washington State Energy Code
2. Optionally upload construction plan images for detailed analysis
3. If images are uploaded, click "Generate Labor Estimate" for automated calculations
4. Continue the conversation to discuss the code or get clarification on estimates
5. Use the "Clear Chat History" button in the sidebar to start fresh

Example questions you can ask:
- What are the key requirements for residential building envelopes?
- How do I ensure compliance with ventilation standards?
- What are the minimum insulation requirements?
- Can you explain the lighting power allowance calculations?
- What are the requirements for mechanical systems?
""")

# Download button for chat history
if st.session_state.messages:
    chat_history = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.messages])
    st.download_button(
        "Download Chat History",
        chat_history,
        file_name="wsec_chat_history.txt",
        mime="text/plain"
    )