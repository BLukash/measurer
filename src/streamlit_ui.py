import streamlit as st
from PIL import Image, ImageDraw
import random
from src.scale_detector.scale_detector import ScaleDetector


def draw_segment(image: Image, segment):
    draw = ImageDraw.Draw(image)
    draw.line([(segment.a.x, segment.a.y), (segment.b.x, segment.b.y)], fill='red', width=5)
    return image

def main():
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f8f9fa;
            padding: 10px;
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="footer">© Bohdan Lukashchuk. All rights reserved.</div>',
        unsafe_allow_html=True
    )


    st.title("ScaleSense")
    st.markdown("""Welcome to ScaleSense, an innovative tool for identifying the real-world scale on your images. By detecting the scale of a ruler present in an image, ScaleSense effectively translates pixels into real-world units, providing a precise measure of 1cm in pixels.

While the primary application is medical imaging, where accurate dimensions are critical, the technology can also be used for other purposes such as forensics, museum artifact measurements, and more.

Ready to try? Select one of our sample images or upload your own, and let ScaleSense do the rest! (do not forget to check image caption after))""")

    st.markdown("## Original Publication")
    st.markdown("[Link to the original publication](https://ieeexplore.ieee.org/document/9916468)")
    st.markdown("Current demo is based on the second version of the ScaleSense method and link to the ScaleSenseV2 paper will be published soon)")

    grouping_options = {
        'DBSCAN': 'cluster', 
        'Modified Z-score': 'z-score', 
        'Interquartile Range (IQR)': 'IQR', 
        'Median': 'median'
    }

    st.markdown("""
    This dropdown allows you to select the method for grouping multiple candidate ruler segments, before final ruler scale detection.
    We propose you to play with these options!)
    """)
    grouping = st.selectbox("Grouping method", list(grouping_options.keys()))

    dbscan_args = {}
    grouping_method = grouping_options[grouping]
    if grouping_method == 'cluster':
        dbscan_args['eps'] = st.number_input('DBSCAN eps', value=20)
        dbscan_args['min_samples'] = st.number_input('DBSCAN min_samples', value=1)


    st.markdown("""
    :warning: **Sensitive Content Warning** :warning:
    The following section contains medical images that some individuals may find disturbing.
    Viewer discretion is advised.
    """)

    with st.expander("Click Here to View Sample Images to Test Method on", expanded=False):
        sample_images = {"Sample Image 1": "./datasets/ruler_digit/images/all/DSC_0087.JPG",
                        "Sample Image 2": "./datasets/ruler_digit/images/all/IMG_1601.JPG",
                        "Sample Image 3": "./datasets/ruler_digit/images/all/DSC_0611.JPG"}

        # Create 3 columns
        col1, col2, col3 = st.columns(3)

        # Display each image in a separate column
        sample_file = None
        with col1:
            st.image(Image.open(sample_images["Sample Image 1"]))
            if st.button("Use Sample Image 1"):
                sample_file = open(sample_images["Sample Image 1"], 'rb')

        with col2:
            st.image(Image.open(sample_images["Sample Image 2"]))
            if st.button("Use Sample Image 2"):
                sample_file = open(sample_images["Sample Image 2"], 'rb')

        with col3:
            st.image(Image.open(sample_images["Sample Image 3"]))
            if st.button("Use Sample Image 3"):
                sample_file = open(sample_images["Sample Image 3"], 'rb')

    st.markdown("Or upload your own image for scale detection.")
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

    if uploaded_file or sample_file:
        uploaded_file = uploaded_file or sample_file
        img = Image.open(uploaded_file)
        # st.image(img, caption='Uploaded Image', use_column_width=True)
        # Use the selected grouping method from the dictionary
        model_type='v7'

        candidate_segments, pixel_len, all_numbers, numbers_on_one_line = \
            ScaleDetector.detect_ruler_scale(img, grouping_method, model_type, dbscan_args=dbscan_args)

        if pixel_len:
            chosen_segment = random.choice(candidate_segments)
            img = draw_segment(img, chosen_segment)
            st.image(img, use_column_width=True)
            st.markdown(f'### Processed Image. Scale: {pixel_len}px/cm',)
        else:
                st.write("""
                Unfortunately, ScaleSense was not able to detect the ruler scale on this image.
                
                To help us improve the method, we kindly ask you to share this data with us.
                
                We appreciate your contribution and patience. Thank you!
                """)

    st.markdown("""
        # Collaboration
        We are actively looking for collaborations and partnerships. If you have ideas on how to improve our method, 
        are interested in collaborating with us, or have datasets that you believe could help enhance ScaleSense, 
        we would love to hear from you.

        Please feel free to reach out using the contact information below.
        - [LinkedIn](https://www.linkedin.com/in/blukash/)
        - Email: [bohdan.lukashchuk@gmail.com](bohdan.lukashchuk@gmail.com)
    """)

    st.markdown("""
    ## Support the project
    If you like this project and want to support it, please [Buy me a coffee](https://bmc.link/blukash).
    """)

if __name__ == '__main__':
    main()
