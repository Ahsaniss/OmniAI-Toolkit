import streamlit as st
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import json
import PyPDF2
import cv2
from PIL import Image
import logging
import sympy as sp
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = "AIzaSyAF6rrJQCuU8k6uHBAm65wn3M43rtJTpsI"

# Configure the Gemini AI model
genai.configure(api_key=api_key)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chatbot_history" not in st.session_state:
    st.session_state.chatbot_history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "chatbot_response" not in st.session_state:
    st.session_state.chatbot_response = ""

# Helper functions
def clear_cache():
    st.cache_data.clear()
    st.success("Cache cleared successfully!")

def evaluate_equation(equation, x_vals):
    try:
        expr = sp.sympify(equation)
        return [float(expr.subs(sp.Symbol('x'), x_val)) for x_val in x_vals]
    except Exception as e:
        logging.error(f"Error in equation evaluation: {e}")
        st.error(f"Error: Unable to evaluate the equation. {str(e)}")
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
    elif "equation" in parsed_data:
        equation = parsed_data["equation"]
        x = np.linspace(-10, 10, 400)
        y = evaluate_equation(equation, x)
        if y:
            ax.plot(x, y, label=f"{equation}")
    
    ax.legend()
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
    Parse the following input and return a JSON object.
    If it contains points, return as: {{"points": [[x1, y1], [x2, y2], ...]}}.
    If it contains an equation, return as: {{"equation": "equation_in_terms_of_x"}}.
    Input: {user_input}
    """
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        parsed_data = json.loads(response.text)
        return parsed_data
    except json.JSONDecodeError:
        st.error("Error: Unable to parse Gemini's response. Please check the format.")
        return {}
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return {}

def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def dodgeV2(x, y):
    return cv2.divide(x, 255 - y, scale=256)


def generate_response(user_input):
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        You are a friendly and knowledgeable chatbot. 
        Provide detailed and engaging responses to the user's queries. 
        Always be polite and encouraging.
        User: {user_input}
        Chatbot:
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again later."

def clear_chat_state():
    st.session_state.user_input = ""
    st.session_state.chatbot_response = ""

# Default Page with Instructions
def show_default_page():
    st.title("Welcome to the OmniAI Toolkit!")
    st.write("This app provides several exciting features. Use the sidebar to select a feature you want to explore.")
    
    st.markdown("""
    ## How to Use This App
    - **Graph Making AI**: Enter a mathematical expression or points to generate graphs.
    - **Searchable Document Chatbot**: Upload a PDF and ask questions about its content.
    - **Chatbot**: Interact with an AI chatbot to answer your questions.
    - **Pencil Sketch Converter**: Upload an image and convert it to a pencil sketch.
    - **Summary Generator**: Enter text and generate a summary of it.
    
    Explore these features from the sidebar!
    """)

    st.markdown("""
    <div style='text-align: center; margin-top: 50px;'>
        <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSzcgQ6toDLgGTJzH1wY5AjqR0Zk38RBLU7TA&s' width='80' />
        <p>Powered by Ahsan TECH</p>
    </div>
    """, unsafe_allow_html=True)

# Apply Custom CSS for Background Colors
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .header {
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        background-color: #f0f2f6;
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stApp {
        background-color: #f7f7f7;
        padding: 10px;
    }
    .sidebar .sidebar-content {
        background-color:#f0f2f6;
        color: #f0f2f6;
    }
    .stButton>button {
        background-color: #6200ea;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Main App
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Select a feature",
    ("Default Page", "Graph Making AI", "Searchable Document Chatbot", "Chatbot",  "Summary Generator")
)

if option == "Default Page":
    show_default_page()

elif option == "Graph Making AI":
    st.markdown("""
        <style>
        .header {
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            background-color: #6200ea;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        </style>
        <div class="header">Graph Generator AI</div>
        """, unsafe_allow_html=True)

    user_input = st.text_input("Enter a mathematical question (e.g. 'Plot the points (1, 2) and (3, 4)' or 'y = 2x + 3'):")
    x_unit = st.text_input("Enter the unit for the X axis (e.g., time, meters, etc.):", value="units")
    y_unit = st.text_input("Enter the unit for the Y axis (e.g., distance, millions, etc.):", value="units")

    if st.button("Generate Graph"):
        if user_input:
            parsed_data = process_input_with_gemini(user_input)
            if parsed_data:
                plot_graph(parsed_data, x_unit, y_unit)
                explanation = generate_response(f"Explain how to plot the graph for: {user_input}")
                st.markdown("### Explanation")
                st.write(explanation)
            else:
                st.error("Unable to process the input. Please try rephrasing your question.")

    if st.button("Clear Cache"):
        clear_cache()

elif option == "Searchable Document Chatbot":
    
    st.subheader("🔍 Searchable Document Chatbot")
    uploaded_file = st.file_uploader("Choose your .pdf file", type="pdf")

    if uploaded_file:
        st.subheader("📄 PDF Preview")
        try:
            page_count = len(PyPDF2.PdfReader(uploaded_file).pages)
            st.write(f"Total pages: {page_count}")
            st.image("https://via.placeholder.com/150", caption="PDF Preview")

            document_text = extract_text_from_pdf(uploaded_file)
            if document_text:
                st.success("✅ PDF loaded successfully!")
                st.subheader("📜 PDF Document Preview")
                st.text_area("Document Content", document_text[:5000], height=300)

                user_query = st.text_input("Ask questions about the document:")

                if st.button("Generate Answer"):
                    st.write("Generating...")
                    # Process the user query with AI
                    
                    answer = generate_response(user_query)
                    st.subheader("🤖 AI Answer")
                    st.write(answer)

        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    
    if st.button("Clear Chat"):
        clear_chat_state()

elif option == "Chatbot":
    
    st.subheader("🤖 Chatbot")
    user_input = st.text_input("You: ", key="user_input")
    if st.button("Send"):
        chatbot_response = generate_response(user_input)
        st.write(f"AI: {chatbot_response}")
        
  

elif option == "Summary Generator":
    
    st.subheader("📝 Summary Generator")
    user_text = st.text_area("Enter text to summarize:")

    if st.button("Generate Summary"):
        summary = generate_response(f"Summarize the following text: {user_text}")
        st.subheader("Summary")
        st.write(summary)
