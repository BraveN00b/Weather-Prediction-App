import streamlit as st
from PIL import Image

st.set_page_config(page_title="Weather Prediction App Project", page_icon=":sun_behind_rain_cloud:")

# Load Assets
img_darshpreet = Image.open("photos/darshpreet_photo.jpg")
img_jaskaran = Image.open("photos/jaskaran_photo.jpg")
img_abhayjit = Image.open("photos/abhayjit_photo.jpg")

# Main Content
st.header("Weather Prediction App Project")

st.subheader("NITTTR Chandigarh")

st.write("---")
# Introductory paragraph
st.markdown("**Task Introduction:**")
st.write("""Over the course of last two weeks, we undertook a project to create a Weather Prediction App that employs data science and machine learning techniques to forecast weather conditions based on user-input parameters. This project was a part of our comprehensive data science course, aimed at equipping us with practical skills in machine learning, data preprocessing, and application development.""")

st.markdown("**Learning Process:**")
st.write("""Throughout the project, we harnessed the knowledge we gained from the course to construct an accurate weather prediction model using the XGBoost algorithm. We applied data preprocessing techniques, learned about feature scaling, and evaluated our model's performance using various metrics. Additionally, we dived into the world of Streamlit, leveraging its capabilities to build an intuitive user interface for the app, allowing users to easily input parameters and obtain predictions.""")

# Contributors and their photos
st.subheader("Developed by:")

# Use a container for layout control
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    
    # Left column
    with left_column:
        st.image(img_darshpreet, width=200)
    
    # Right column
    with right_column:
        st.markdown("- **Darshpreet Singh**")
        st.markdown("Data preprocessing")

with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    
    # Left column
    with left_column:
        st.image(img_jaskaran, width=200)
    
    # Right column
    with right_column:
        st.markdown("- **Jaskaran Singh**")
        st.markdown("Machine learning model and Project coordination")

with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    
    # Left column
    with left_column:
        st.image(img_abhayjit, width=200)
    
    # Right column
    with right_column:
        st.markdown("- **Abhayjit Singh**")
        st.markdown("User interface design and Streamlit implementation")


st.write("---")
st.write("""In summary, our journey through the six-week data science course culminated in the development of the Weather Prediction App. By seamlessly merging theoretical knowledge with practical application, we not only expanded our skill set but also produced an interactive and insightful tool that combines the realms of data science and user-oriented design.""")