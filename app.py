import os
import streamlit as st
import google.generativeai as genai
import PIL.Image
from dotenv import find_dotenv, load_dotenv
from langchain.schema import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv(find_dotenv(), override=True)

# Streamlit app title
st.title("Give me an image of food I will give you the recipe")

# Sidebar for API key input
api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)

# Configure the generation parameters
generation_config = {'temperature': 0.9}

# File uploader for the image
uploaded_file = st.file_uploader("Upload an image of Food", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using PIL
    img = PIL.Image.open(uploaded_file)

    # Display the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Basic heuristic to check if the image might contain food
    # This is a simple check and can be improved with actual image classification models
    # For demonstration purposes, we will check the image size and some colors
    img_size = img.size
    is_food_image = (img_size[0] > 100 and img_size[1] > 100)  # Check if image dimensions are reasonable

    # If it's not likely a food image, show a warning
    if not is_food_image:
        st.warning("This image does not seem to be food. Please select an image of food only for the recipe.")
    else:
        # Generate content from the image
        model = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config)
        with st.spinner("Generating content from the image..."):
            response = model.generate_content(img)

        # Extract dish name and ingredients
        prompt = f'Extract dish name and main ingredients from this description: {response}'
        with st.spinner("Extracting dish name and ingredients..."):
            response = model.generate_content(prompt)

        # Display the extracted information
        st.subheader("Extracted Information")
        st.write(response.text)

        # ChatGPT-like recipe response
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.9)
        message = [
            SystemMessage(content="You are a chef, and you have to provide a detailed recipe and country name."),
            HumanMessage(content=response.text)
        ]
        
        with st.spinner("Generating recipe..."):
            recipe = llm.invoke(message)

        # Display the recipe content
        st.subheader("Generated Recipe")
        st.write(recipe.content)
