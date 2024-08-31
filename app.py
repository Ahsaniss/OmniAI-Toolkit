import streamlit as st
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import json
import logging
import sympy as sp
import PyPDF2
from PIL import Image
import cv2

# Configure API key
api_key = "AIzaSyBFCRAdNSHw6894aq_56iBNfaDAhZgsIXI"
genai.configure(api_key=api_key)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Helper Functions
def clear_cache():
    st.cache_data.clear()
    st.success("Cache cleared successfully!")

def evaluate_equation(equation, x_vals):
    try:
        expr = sp.sympify(equation)
        return [float(expr.subs(sp.Symbol('x'), x_val)) for x_val in x_vals]
    except Exception as e:
        logging.error(f"Error in equation evaluation: {e}")
        st.error("Error: Unable to evaluate the equation.")
        return []

def plot_graph(parsed_data, x_unit, y_unit):
    fig, ax = plt.subplots()
    ax.set_xlabel(f"X ({x_unit})")
    ax.set_ylabel(f"Y ({y_unit})")
    ax.set_title("Graph Based on Provided Data")

    if "points" in parsed_data:
        points = parsed_data["points"]
        x_vals = [point[0] for point in points]
        y_vals = [point[1] for point in points]
        ax.plot(x_vals, y_vals, 'bo-', label="Points")
        st.pyplot(fig)

    elif "equation" in parsed_data:
        equation = parsed_data["equation"]
        x = np.linspace(-10, 10, 400)
        y = evaluate_equation(equation, x)
        if y:
            ax.plot(x, y, label=f"{equation}")
            st.pyplot(fig)

def process_input_with_gemini(user_input):
    known_equations = {
        "first equation of motion": "u + a*x",
        "second equation of motion": "u*x + 0.5*a*x**2",
        "third equation of motion": "u**2 + 2*a*x"
    }
    for key, equation in known_equations.items():
        if key in user_input.lower():
            return {"equation": equation}
    prompt = f"""
    Please parse the following input and return a JSON object.
    If it contains points, return as: {{"points": [[x1, y1], [x2, y2], ...]}}.
    If it contains an equation, return as: {{"equation": "equation_in_terms_of_x"}}.
    Input: {user_input}
    """
    response = genai.generate_text(prompt=prompt)
    try:
        parsed_data = json.loads(response.result)
        return parsed_data
    except json.JSONDecodeError:
        st.error("Error: Unable to parse Gemini's response. Please check the format.")
        return {}

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    return text

def dodgeV2(x, y):
    return cv2.divide(x, 255 - y, scale=256)

def pencilsketch(inp_img):
    img_gray = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(img_gray)
    img_smoothing = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0)
    final_img = dodgeV2(img_gray, img_smoothing)
    return final_img

def initialize_chatbot():
    if "chatbot_history" not in st.session_state:
        st.session_state.chatbot_history = []

def generate_response(user_input):
    prompt = f"""
    You are a friendly and knowledgeable chatbot. 
    Provide detailed and engaging responses to the user's queries. 
    If the user asks for something specific, try to provide useful and informative answers. 
    Always be polite and encouraging. 
    User: {user_input}
    Chatbot:
    """
    response = genai.generate_text(prompt=prompt)
    return response.result

# Default Page
def show_default_page():
    st.title("Welcome to the OmniAI Toolkit !")
    st.write("This app provides several exciting features. Use the sidebar to select a feature you want to explore.")
    st.markdown("""
    <div style='text-align: center; margin-top: 50px;back-ground-color:#07070d';>
        <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSzcgQ6toDLgGTJzH1wY5AjqR0Zk38RBLU7TA&s' width='80' />
        <p>Powered by Ahsan TECH</p>
    </div>
    """, unsafe_allow_html=True)

# Main App
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Select a feature",
    ("Default Page", "Graph Making AI", "Searchable Document Chatbot", "Chatbot", "Pencil Sketch (Feature 4)", "Summary Generator")
)

if option == "Default Page":
    show_default_page()

elif option == "Graph Making AI":
    st.markdown("""
        <style>
        .header {
            font-size: 2em;
            font-weight: bold;
            text-align: center;
            background-color: #f7f7f7;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        </style>
        <div class="header">Graph Generator AI <img src="https://cdn-icons-png.flaticon.com/512/189/189001.png" width="40" /></div>
        """, unsafe_allow_html=True)

    user_input = st.text_input("Enter a mathematical question (e.g. 'Plot the points (1, 2) and (3, 4)' or 'y = 2x + 3'):")
    x_unit = st.text_input("Enter the unit for the X axis (e.g., time, meters, etc.):", value="units")
    y_unit = st.text_input("Enter the unit for the Y axis (e.g., distance, millions, etc.):", value="units")

    if st.button("Search"):
        if user_input:
            parsed_data = process_input_with_gemini(user_input)
            if parsed_data:
                plot_graph(parsed_data, x_unit, y_unit)
            explanation = genai.generate_text(prompt=f"Explain how to plot the graph for: {user_input}")
            st.markdown("### Explanation")
            st.write(explanation.result)

    if st.button("Clear Cache"):
        clear_cache()

    st.markdown("""
        <div style='text-align: center; margin-top: 50px;'>
            <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSzcgQ6toDLgGTJzH1wY5AjqR0Zk38RBLU7TA&s' width='80' />
            <p>Powered by Ahsan TECH</p>
        </div>
        """, unsafe_allow_html=True)

elif option == "Searchable Document Chatbot":
    st.subheader("🔍 Searchable Document Chatbot")
    uploaded_file = st.file_uploader("Choose your .pdf file", type="pdf")

    if uploaded_file:
        st.subheader("📄 PDF Preview")
        page_count = len(PyPDF2.PdfReader(uploaded_file).pages)
        st.write(f"Page 1 of {page_count}")
        st.image("https://via.placeholder.com/150", caption="PDF Preview")

        st.success("✅ Ready to Chat!")
        document_text = extract_text_from_pdf(uploaded_file)
        st.subheader("📜 PDF Document Preview")
        st.text_area("Document Content", document_text[:5000], height=300)

        user_question = st.text_input("💬 Ask a question")
        if st.button("Search"):
            if user_question:
                chat_session = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    generation_config={
                        "temperature": 1,
                        "top_p": 0.95,
                        "top_k": 64,
                        "max_output_tokens": 8192,
                        "response_mime_type": "text/plain",
                    },
                ).start_chat(history=[])
                response = chat_session.send_message(f"Document: {document_text}\n\nQuestion: {user_question}")
                st.subheader("🤖 Chatbot Response")
                st.markdown(f"""
                <div style='background-color:#f9f9f9; padding:10px; border-radius:5px;'>
                    <strong>User Question:</strong> {user_question} <br><br>
                    <strong>AI Response:</strong> {response.text}
                </div>
                """, unsafe_allow_html=True)

        if st.button("Clear Chat"):
            st.cache_data.clear()

    else:
        st.info("💡 Upload a PDF to start the chat!")

    st.markdown("---")
    st.write(
        "Made with ❤️ by Ahsan TECH"
    )

elif option == "Chatbot":
    st.subheader("🤖 Chatbot")

    # Initialize chatbot
    initialize_chatbot()

    # Chat history display
    if st.session_state.chatbot_history:
        for entry in st.session_state.chatbot_history:
            st.write(f"**User:** {entry['user']}")
            st.write(f"**Chatbot:** {entry['response']}")
    
    # User input
    user_message = st.text_input("Type your message here...", "")
    if st.button("Send"):
        if user_message:
            response_text = generate_response(user_message)
            st.session_state.chatbot_history.append({"user": user_message, "response": response_text})
            st.write(f"**Chatbot:** {response_text}")
        else:
            st.warning("Please type a message before sending.")

    st.markdown("""
        <div style='text-align: center; margin-top: 50px;'>
            <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSzcgQ6toDLgGTJzH1wY5AjqR0Zk38RBLU7TA&s' width='80' />
            <p>Powered by Ahsan TECH</p>
        </div>
        """, unsafe_allow_html=True)

elif option == "Pencil Sketch (Feature 4)":
    st.title("Pencil Sketch Converter")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Convert image to pencil sketch
        image = Image.open(uploaded_image).convert("RGB")
        img_np = np.array(image)
        pencil_sketch = pencilsketch(img_np)
        st.image(pencil_sketch, caption="Pencil Sketch", use_column_width=True)

    st.markdown("""
        <div style='text-align: center; margin-top: 50px;'>
            <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSzcgQ6toDLgGTJzH1wY5AjqR0Zk38RBLU7TA&s' width='80' />
            <p>Powered by Ahsan TECH</p>
        </div>
        """, unsafe_allow_html=True)

elif option == "Summary Generator":
    st.title("Summary Generator")

    # Text area for input
    user_text = st.text_area("Enter the text to summarize:", "")

    if st.button("Generate Summary"):
        if user_text:
            prompt = f"Summarize the following text:\n\n{user_text}"
            response = genai.generate_text(prompt=prompt)
            st.subheader("Summary:")
            st.write(response.result)
        else:
            st.warning("Please enter some text to summarize.")

    st.markdown("""
        <div style='text-align: center; margin-top: 50px;'>
            <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSzcgQ6toDLgGTJzH1wY5AjqR0Zk38RBLU7TA&s' width='80' />
            <p>Powered by Ahsan TECH</p>
        </div>
        """, unsafe_allow_html=True)
