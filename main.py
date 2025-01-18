import streamlit as st
from PIL import Image
import google.generativeai as genai
from gemini_helper import GeminiEstimator

# Page configuration
st.set_page_config(
    page_title="Framing Labor Estimator",
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
if 'mode' not in st.session_state:
    st.session_state.mode = 'chat'  # Default to chat mode

# Sidebar for API key and mode selection
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    if api_key:
        estimator = GeminiEstimator(api_key)
    else:
        estimator = GeminiEstimator()
    
    # Mode selection
    st.session_state.mode = st.radio("Select Mode", ['Chat', 'Labor Estimation'])
    
    # Add clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat = estimator.start_chat()
        st.session_state.images_analyzed = False
        st.session_state.current_images = []
        st.rerun()

# Initialize chat if not already done
if st.session_state.chat is None:
    st.session_state.chat = estimator.start_chat()

# Main interface based on mode
if st.session_state.mode == 'Labor Estimation':
    st.title("🏗️ Framing Labor Estimator")
    st.markdown("Upload construction plan images to automatically estimate framing labor hours and costs.")
    
    # File uploader
    uploaded_files = st.file_uploader("Upload construction plan images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    
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
                        st.session_state.current_images = [Image.open(file) for file in uploaded_files]
                        report = estimator.analyze_images(st.session_state.current_images, st.session_state.chat)
                        st.session_state.messages.append({"role": "assistant", "content": report})
                        st.session_state.images_analyzed = True
                        st.rerun()

else:  # Chat mode
    st.title("💬 Construction Assistant Chat")
    st.markdown("Ask questions about construction, framing, or general building topics.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..." if st.session_state.mode == 'Chat' else "Ask questions about the labor estimate..."):
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
if st.session_state.mode == 'Labor Estimation':
    st.markdown("""
    ### How to Use This Framing Labor Estimator
    1. Upload clear images of your construction plans (multiple images supported)
    2. Click "Generate Labor Estimate" to get an automated calculation
    3. Review the detailed breakdown of labor hours and costs
    4. Ask questions about specific components or calculations
    5. Use the "Clear Chat History" button in the sidebar to start fresh

    Example questions you can ask:
    - Can you break down the wall framing calculations in more detail?
    - What assumptions were made about the roof system?
    - How would weather conditions impact this estimate?
    - Can you explain the crew size recommendation?
    - What factors contributed to the contingency calculations?
    """)
else:
    st.markdown("""
    ### How to Use the Construction Assistant Chat
    - Ask general questions about construction and building
    - Get advice on framing techniques and best practices
    - Learn about construction materials and methods
    - Discuss project planning and management
    - Get explanations about building codes and regulations
    
    Example questions you can ask:
    - What are the key considerations for framing a load-bearing wall?
    - How do I calculate the proper spacing for floor joists?
    - What's the difference between platform and balloon framing?
    - Can you explain the basics of roof truss design?
    - What safety measures should be considered for framing work?
    """)

# Download button for chat history
if st.session_state.messages:
    chat_history = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.messages])
    st.download_button(
        "Download Chat History",
        chat_history,
        file_name="construction_chat.txt" if st.session_state.mode == 'Chat' else "labor_estimate.txt",
        mime="text/plain"
    )