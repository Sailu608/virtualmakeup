import sqlite3
import bcrypt
import cv2
import numpy as np
from skimage.filters import gaussian
from test import evaluate
import streamlit as st
from PIL import Image, ImageColor

# Function to add background image
def add_bg_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover; 
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
def side_bg_image(image_url):
    st.markdown(
        f"""
        <style>
        .stSidebar{{
            background-image: url({image_url});
            background-size: cover; 
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# Function to add custom text styles
def add_custom_text_styles():
    st.markdown(   
        """
        <style>
        h1 {
            color: black;  
            font-family: 'Arial', sans-serif;          
            font-size: 50px;
            text-align: center;
            letter-spacing: 2px;
        }
        h2 {
            color: #2E8B57;
            font-family: 'Verdana', sans-serif;
            font-size: 35px;
            text-align: left;
        }
        h3 {
            color:rgb(53, 131, 37);
            font-weight: bold;
            font-size: 35px;
        }
        p {
            font-size: 25px;
            font-weight: bold;   
            color:rgb(219, 32, 131);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Database functions (signup, login)
def create_db():
    """Creates the users table in the SQLite database if it doesn't exist."""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()


def register_user(username, email, password):
    """Registers a new user in the database with a hashed password."""
    if isinstance(password, str):
        password = password.encode('utf-8')

    hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())  # Hash the password

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO users (username, email, password)
            VALUES (?, ?, ?)
        ''', (username, email, hashed_password))
        conn.commit()
    except sqlite3.IntegrityError:
        return False  # Username already exists
    finally:
        conn.close()
    return True


def verify_user(username, password):
    """Verifies a user's credentials."""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username=?', (username,))
    user_data = c.fetchone()
    conn.close()

    if user_data:
        stored_hash = user_data[0]

        if isinstance(password, str):
            password = password.encode('utf-8')  # Convert to bytes if it's a string

        if bcrypt.checkpw(password, stored_hash):
            return True

    return False


# Makeup Functions
def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, channel_axis=-1)
    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img
    img_out = img_out / 255.0
    mask_1 = img_out < 0
    mask_2 = img_out > 1
    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r = color
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r
    np.repeat(parsing[:, :, np.newaxis], 3, axis=2)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part == 12 or part == 13:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    if part == 17:
        changed = sharpen(changed)
    changed[parsing != part] = image[parsing != part]
    return changed


# UI Functions
def signup_page():
    add_custom_text_styles()
    add_bg_image('https://static.vecteezy.com/system/resources/previews/037/246/944/non_2x/ai-generated-make-up-products-advertisment-background-with-copy-space-free-photo.jpg')
    st.title("Signup")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type='password')
    confirm_password = st.text_input("Confirm Password", type='password')

    if password == confirm_password:
        if st.button("Create Account"):
            if register_user(username, email, password):
                st.success("Account created successfully!")
                st.write("Go to [Login page](#login) to log in.")
            else:
                st.error("Username already exists!")
    else:
        st.error("Passwords do not match!")


def login_page():
    add_custom_text_styles()
    add_bg_image('https://static.vecteezy.com/system/resources/previews/037/246/944/non_2x/ai-generated-make-up-products-advertisment-background-with-copy-space-free-photo.jpg')
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')

    if st.button("Login"):
        if verify_user(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success(f"Welcome, {username}!")
            st.write("Redirecting to the Virtual Makeup app...")
            st.rerun()  # Reload to show makeup page after login
        else:
            st.error("Invalid username or password")

def show_makeup_app():
    st.markdown(
    """
    <h1 style="display: flex; align-items: center;">
        <img src="https://banner2.cleanpng.com/20180202/sew/av2lgx1ek.webp" width="100" height="70" style="margin-right: 20px;">
        Virtual Makeup 💄
    </h1>
    """, unsafe_allow_html=True)
    add_custom_text_styles()
    add_bg_image('https://i.pinimg.com/736x/82/77/af/8277af0d598eaffa1deef050fec18ee5.jpg')

    table = {
        'hair': 17,
        'upper_lip': 12,
        'lower_lip': 13,
    }

    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Check if a file is uploaded, otherwise use a default image
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
        demo_image = img_file_buffer  # Update demo_image if the file is uploaded
    else:
        demo_image = 'imgs/116.jpg'  # Default image path
        image = np.array(Image.open(demo_image))  # Load default image

    st.subheader('Original Image')
    st.image(image, use_column_width=True)
    
    cp = 'cp/79999_iter.pth'
    ori = image.copy()
    h, w, _ = ori.shape

    image = cv2.resize(image, (1024, 1024))
    parsing = evaluate(demo_image, cp)
    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

    parts = [table['hair'], table['upper_lip'], table['lower_lip']]

    hair_color = st.sidebar.color_picker('Pick the Hair Color', '#000')
    hair_color = ImageColor.getcolor(hair_color, "RGB")

    lip_color = st.sidebar.color_picker('Pick the Lip Color', '#edbad1')
    lip_color = ImageColor.getcolor(lip_color, "RGB")

    colors = [hair_color, lip_color, lip_color]

    apply_button = st.sidebar.button("Apply Makeup")

    if apply_button:
        for part, color in zip(parts, colors):
            image = hair(image, parsing, part, color)

        image = cv2.resize(image, (w, h))
        st.subheader('Output Image')
        st.image(image, use_column_width=True)
    else:
        st.write("Press the 'Apply Makeup' button to apply the makeup to the image.")

# Ensure the database is created when the app starts
create_db()

# Check if the user is logged in
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    pages = ["Signup", "Login"]
    page = st.sidebar.selectbox("Choose Page", pages)
    side_bg_image("https://i.pinimg.com/736x/82/77/af/8277af0d598eaffa1deef050fec18ee5.jpg")
    if page == "Signup":
        signup_page()
    else:
        login_page()
else:
    # If the user is logged in, show the makeup app
    show_makeup_app()
