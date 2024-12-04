import streamlit as st
from datetime import datetime
from PIL import Image, ImageEnhance
import numpy as np
import tensorflow as tf
from rembg import remove
from io import BytesIO
import cv2
import base64
from recond import *
from recond_hindi import *
from recond_marathi import *
from ogin import *

# Load your pre-trained models for various crops
cnn_tomato = tf.keras.models.load_model('trained_plant_disease_model_tomato.keras')
cnn_strawberry = tf.keras.models.load_model('trained_plant_disease_model_strobery.keras')
cnn_apple = tf.keras.models.load_model('trained_plant_disease_model_apple.keras')
cnn_potato = tf.keras.models.load_model('trained_plant_disease_model_potato.keras')
cnn_peach = tf.keras.models.load_model('trained_plant_disease_model_peach.keras')
cnn_pepperbell = tf.keras.models.load_model('trained_plant_disease_model_paperbell.keras')
cnn_corn = tf.keras.models.load_model('trained_plant_disease_model_corn.keras')
cnn_grape = tf.keras.models.load_model('trained_plant_disease_model_grape.keras')

# Class names for different crops
class_name_strawberry = ['Strawberry - Leaf_scorch', 'Strawberry - healthy']
class_name_apple = ['Apple | Apple_scab', 'Apple | Black_rot', 'Apple | Cedar_apple_rust', 'Apple | healthy']
class_name_potato = ['Potato | Early_blight', 'Potato | Late_blight', 'Potato | healthy']
class_name_tomato = ['Tomato | Bacterial_spot', 
                     'Tomato | Early_blight', 
                     'Tomato | Late_blight', 
                     'Tomato | Leaf_Mold', 
                     'Tomato | Septoria_leaf_spot', 
                     'Tomato | Spider_mites Two-spotted_spider_mite', 
                     'Tomato | Target_Spot', 
                     'Tomato | Tomato_Yellow_Leaf_Curl_Virus', 
                     'Tomato | Tomato_mosaic_virus', 
                     'Tomato | healthy']
class_name_peach = ['Peach | Bacterial_spot', 'Peach | healthy']
class_name_corn = ['Corn_(maize) | Cercospora_leaf_spot Gray_leaf_spot', 
                   'Corn_(maize) | Common_rust_', 
                   'Corn_(maize) | Northern_Leaf_Blight', 
                   'Corn_(maize) | healthy']
class_name_grape = ['Grape | Black_rot', 'Grape | Esca_(Black_Measles)', 'Grape | Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape | healthy']
class_name_pepperbell = ['Pepper,_bell | Bacterial_spot', 'Pepper,_bell | healthy']

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)), 
            url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
def get_image_base64(img_path):
    # Convert the image to base64 format
    with open(img_path, "rb") as img_file:
        img_bytes = img_file.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64
def home_page():
    # Add background image from local file with transparency
    add_bg_from_local('back1.jpg')

    # Title and introduction with default Streamlit margins and padding
    st.markdown("""  
        <div style='text-align: center; padding: 20px; 
                    background: rgba(255, 255, 255, 0.7); 
                    margin-bottom: 40px; width: 100%;'>
            <h1 style='color: #2c3e50;'>Crop Disease Prediction App</h1>
        </div>
    """, unsafe_allow_html=True)

    # Layout for columns with increased width
    col1, col2 = st.columns([3, 3], gap="large")  # Define relative column widths

    with col1:
        img_base64 = get_image_base64("back1.jpg")
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/jpeg;base64,{img_base64}" alt="Home Image" 
                     style="width: 100%; max-width: 100%; height: 400px; border-radius: 10px;">
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            '''
            <div style='text-align: justify; color:black; font-size: 22px; width: 100%;'>
                Our app offers a comprehensive solution for plant health management by integrating 
                a plant disease detector and medicine suggestor in one platform. It uses advanced image recognition technology to identify plant diseases from photos you take, providing instant, accurate diagnoses. 
                Based on the detected disease, the app then recommends tailored treatment options, including chemical treatments, biological controls, and organic remedies. 
                This seamless integration ensures you receive timely and effective solutions to manage plant health, helping you keep your garden or crops in optimal condition.
            </div>
            ''',
            unsafe_allow_html=True
        )
@st.cache_resource
def process_and_rotate_grape_image(image_file):
    try:
        pil_image = Image.open(image_file)
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced_image = enhancer.enhance(1)
        img_byte_arr = BytesIO()
        enhanced_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        output_image = remove(img_byte_arr)
        output_image = Image.open(BytesIO(output_image)).convert("RGBA")
        light_purple = (230, 230, 250, 255)
        background = Image.new('RGBA', output_image.size, light_purple)
        combined_image = Image.alpha_composite(background, output_image)
        processed_image_path = "temp_processed_image.png"
        combined_image.save(processed_image_path, format='PNG')
        image = cv2.imread(processed_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        height, width = edges.shape
        top_half = edges[:height // 2, :]
        bottom_half = edges[height // 2:, :]
        top_edges_count = np.sum(top_half == 255)
        bottom_edges_count = np.sum(bottom_half == 255)
        if top_edges_count > bottom_edges_count:
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
            cv2.imwrite(processed_image_path, rotated_image)
        else:
            cv2.imwrite(processed_image_path, image)
        return processed_image_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
@st.cache_resource
def remove_background(image_file):
    try:
        pil_image = Image.open(image_file)
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced_image = enhancer.enhance(1)
        img_byte_arr = BytesIO()
        enhanced_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        output_image = remove(img_byte_arr)
        output_image = Image.open(BytesIO(output_image)).convert("RGBA")
        light_purple = (230, 230, 250, 255)
        background = Image.new('RGBA', output_image.size, light_purple)
        combined_image = Image.alpha_composite(background, output_image)
        img_byte_arr = BytesIO()
        combined_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
@st.cache_resource
def predict_disease(image_file, crop_model, class_names):
    try:
        image = tf.keras.preprocessing.image.load_img(image_file, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        predictions = crop_model.predict(input_arr)
        result_index = np.argmax(predictions)
        return class_names[result_index]
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None
    
def disease_prediction_page():
    add_bg_from_local('back3.jpg')
    # Crops List and their logos
    crops = ['Apple', 'Corn', 'Grape', 'Strawberry', 'Peach', 'Pepperbell', 'Potato', 'Tomato']
    logos = {
        'Apple': 'aplogo.jpg',
        'Corn': 'cornlogo2.jpg',
        'Grape': 'grape.jpeg',
        'Strawberry': 'strobery.jpeg',
        'Peach': 'peach.jpeg',
        'Pepperbell': 'paperbell.jpeg',
        'Potato': 'potatologo2.jpg',
        'Tomato': 'tomatologo.jpg'
    }

    # Initialize selected crop variable
    if 'selected_crop' not in st.session_state:
        st.session_state.selected_crop = None

    # Display a styled and bold title
    st.markdown("""
    <div style='text-align: center; padding: 20px; 
                background: rgba(255, 255, 255, 0.7); 
                margin-bottom: 40px; width: 100%;'>
        <h1 style='color: brown;'>Select A Crop leaf</h1>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4, gap="medium")

    for idx, crop in enumerate(crops):
        with cols[idx % 4]:
            if st.button(crop, key=crop):
                st.session_state.selected_crop = crop
            st.image(logos[crop], use_column_width=True)  
    crop_selected = st.session_state.selected_crop

    if crop_selected:
        st.markdown(f"<h3 style='text-align: center; color: purple;'>Upload an image for {crop_selected} leaf</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: black;'>Upload {crop_selected} image...</h4>", unsafe_allow_html=True)

        # Create the file uploader
        uploaded_image = st.file_uploader("", type=["png", "jpg", "jpeg"])

        if uploaded_image:
            #st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

            # Display message with black color
            st.markdown("<h2 style='color: black;'>Processing image...</h2>", unsafe_allow_html=True)

            # Remove background and rotate image if necessary
            processed_image = None
            try:
                if crop_selected == 'Grape':
                    
                    processed_image_path = process_and_rotate_grape_image(uploaded_image)
                    if processed_image_path:
                        processed_image = processed_image_path
                    else:
                        st.error("Error processing the Grape image.")
                else:
                    processed_image = remove_background(uploaded_image)
                    if not processed_image:
                        st.error("Error removing background for the image.")
            except Exception as e:
                st.error(f"Error during image processing: {e}")

            if processed_image:
                # Map crop type to corresponding model and class names
                
                model_map = {
                    'Tomato': (cnn_tomato, class_name_tomato),
                    'Strawberry': (cnn_strawberry, class_name_strawberry),
                    'Apple': (cnn_apple, class_name_apple),
                    'Potato': (cnn_potato, class_name_potato),
                    'Peach': (cnn_peach, class_name_peach),
                    'Pepperbell': (cnn_pepperbell, class_name_pepperbell),
                    'Corn': (cnn_corn, class_name_corn),
                    'Grape': (cnn_grape, class_name_grape)
                }

                if crop_selected in model_map:
                    model, class_names = model_map[crop_selected]

                    try:
                        
                        disease_prediction = predict_disease(processed_image, model, class_names)

                        if disease_prediction:
                            # Create two columns for layout
                            col1, col2 = st.columns([1, 2], gap="large")  # Column 2 is now twice as wide as Column 1


                            # Column 1: Show Uploaded and Processed Image
                            with col1:
                                st.markdown(f"<h3 style='color: black;'>Uploaded Image:</h3>", unsafe_allow_html=True)
                                st.image(uploaded_image, width=250)  # Reduced width for the uploaded image

                                st.markdown(f"<h3 style='color: black;'>Processed Image:</h3>", unsafe_allow_html=True)
                                st.image(processed_image, width=250)  # Reduced width for the processed image
                            # Column 2: Show Predicted Disease and Recommendation
                            with col2:
                                st.markdown(f"<h3 style='color: purple;'>Predicted Disease:</h3>", unsafe_allow_html=True)
                                st.markdown(f"<span style='color:purple; font-size:24px;font-weight:bold;'>{disease_prediction}</span>", unsafe_allow_html=True)
                                       # Initialize html_content
                                html_content = ""

                                
                                st.markdown("""
                                    <style>
                                    div.stTextInput label {
                                        display: none;  /* Hide the default label */
                                    }
                                    h6 {
                                        margin-bottom: -10px;  /* Adjust negative margin for the label */
                                    }
                                    </style>
                                    <h6 style='color: black;'>Select Language :</h6>
                                """, unsafe_allow_html=True)
                                # Language selection with Marathi option
                                language = st.selectbox("", ["English", "हिंदी", "मराठी"])

                                if st.button("Get Treatment Suggestion"):
                                    recommendation = give_recommendation(disease_prediction)
    
                                       # Check selected language and translate if necessary
                                    if language == "हिंदी":
                                           # Implement this function to translate the recommendation to Hindi
                                           recommendation = give_recommendation_in_hindi(disease_prediction)
                                    elif language == "मराठी":
                                        # Implement this function to translate the recommendation to Marathi
                                        recommendation = give_recommendation_in_marathi(disease_prediction)
                                
                                    # Create HTML content
                                    html_content = f"""
                                       <div style='color: black; text-align: justify; font-size:20px; padding-left:10px;'>
                                           <div style='margin-top: 10px; padding: 15px; border: 2px solid black; 
                                           border-radius: 8px; background-color: rgba(255, 255, 255, 0.9);'>
                                               {recommendation.replace('\n', '<br>')}
                                           </div>
                                       </div>
                                       """
    
                                     # Display the HTML content
                                    st.markdown(html_content, unsafe_allow_html=True)


                                



                        else:
                            st.error("Could not predict the disease.")
                    except Exception as e:
                        st.error(f"Error during disease prediction: {e}")
                else:
                    st.error("Model for the selected crop is not available.")
        else:
            st.markdown("<h5 style='color: red;'>Please upload an image to proceed.</h5>", unsafe_allow_html=True)
    else:
        st.write("Please select a crop.")

def about_us_page():
    add_bg_from_local('back4.jpg')
    st.markdown("<h2 style='text-align: center;'>About Us</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; font-size: 18px;'>
        We are a team dedicated to improving plant health management through advanced technology. Our app leverages the power of machine learning to provide accurate and timely diagnoses of plant diseases. Our goal is to help farmers and gardeners maintain healthy crops and plants, ensuring better yields and sustainable practices.
    </div>
    """, unsafe_allow_html=True)
def help_page():
    add_bg_from_local('back4.jpg')
    st.markdown("<h2 style='text-align: center;'>Help</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size: 18px;'>
        <ul>
            <li><strong>Upload an Image:</strong> Choose an image of the plant leaf showing symptoms of a disease.</li>
            <li><strong>Select a Crop:</strong> Choose the crop type for which the disease needs to be predicted.</li>
            <li><strong>View Results:</strong> The app will process the image and provide the predicted disease along with suggestions for treatment.</li>
            <li><strong>Contact Support:</strong> If you encounter any issues, please contact us through the support page or email us at <a href="mailto:support@example.com">support@example.com</a>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
def main():
    st.set_page_config(page_title="Crop Disease Prediction", page_icon="🌱", layout="centered")

    # Custom CSS for the sidebar
 # Custom CSS for sidebar styling with fixed width
    st.markdown(
    """
    <style>
    /* Sidebar background color with transparency and fixed width */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.7) !important;
        color: white !important;
        width: 200px !important;  /* Set fixed width */
    }
    
    /* Sidebar text color */
    [data-testid="stSidebar"] .css-1v3fvcr {
        color: white !important;
    }

    /* Remove default margin padding */
    .css-18e3th9 {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Center align the sidebar content */
    [data-testid="stSidebar"] .css-1aumxhk {
        text-align: center;
    }

    /* Adjust the main content padding */
    .css-1d391kg {
        padding: 1rem 3rem 1rem 1rem;
    }

    /* Adjust width for the sidebar content */
    [data-testid="stSidebar"] .css-1lcbmhc {
        width: 200px !important;
    }
    
    </style>
    """,
    unsafe_allow_html=True
    )

    # Initialize session state if not already initialized
    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = ''

    # Navigation based on session state
    if st.session_state['logged_in']:
        # Set "Home" as the default page after login
        if st.session_state['page'] == 'login':
            st.session_state['page'] = 'Home'
        
        # Display the sidebar with navigation options
        st.sidebar.title(f"Welcome, {st.session_state['username']}")
        
        if st.sidebar.button("Home"):
            st.session_state['page'] = "Home"
        if st.sidebar.button("Disease Prediction"):
            st.session_state['page'] = "Disease Prediction"
        if st.sidebar.button("About Us"):
            st.session_state['page'] = "About Us"
        if st.sidebar.button("Help"):
            st.session_state['page'] = "Help"
        if st.sidebar.button("Logout"):
            st.session_state['logged_in'] = False
            st.session_state['page'] = 'login'
            st.session_state['username'] = ''
            st.experimental_rerun()  # Refresh to navigate to the login page
        
        # Display the selected page
        if st.session_state['page'] == "Home":
            home_page()
        elif st.session_state['page'] == "Disease Prediction":
            disease_prediction_page()
        elif st.session_state['page'] == "About Us":
            about_us_page()
        elif st.session_state['page'] == "Help":
            help_page()

    else:
        # If not logged in, show the login or registration page
        if st.session_state['page'] == 'login':
            login_page()
        elif st.session_state['page'] == 'register':
            registration_page()
        elif st.session_state['page'] == 'forgot_password':
            forgot_password_page()
    

if __name__ == "__main__":
    main()
