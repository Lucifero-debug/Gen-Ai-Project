import streamlit as st
from few_shot import FewShotPost
from post_generator import generate_post
from scrapper import scrapping
from urllib.parse import urlparse
from preprocessor import main_preprocess

length_options = ["Short", "Medium", "Long"]
language_options = ["English", "Hinglish"]
influencer_options = [
    "warikoo", "kunalshah1", "garyvaynerchuk", "justinwelsh",
    "swati-bathwal-211052143", "sahilbloom", "morgan-housel-5b473821"
]

def main():
    st.subheader("LinkedIn Post Generator")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_influencer = st.selectbox("Choose Influencer", options=influencer_options)

    with col2:
        selected_length = st.selectbox("Length", options=length_options)

    with col3:
        selected_language = st.selectbox("Language", options=language_options)

    custom_influencer_url = st.text_input("Or paste influencer LinkedIn URL", placeholder="https://www.linkedin.com/in/...")

    if "influencer_id" not in st.session_state:
        st.session_state.influencer_id = None
    if "tags" not in st.session_state:
        st.session_state.tags = []

    if custom_influencer_url:
        if st.button('Fetch'):
            with st.spinner("🔍 Scraping LinkedIn posts..."):
                path = urlparse(custom_influencer_url).path
                name = path.split('/')[2]
                scrapping(name)
                st.session_state.influencer_id = name

            with st.spinner("⚙️ Processing content..."):
                main_preprocess(name)
                fs = FewShotPost(name)
                st.session_state.tags = fs.get_tags()

            st.success("✅ Fetch and processing complete!")

    else:
        st.session_state.influencer_id = selected_influencer
        main_preprocess(selected_influencer)
        fs = FewShotPost(selected_influencer)
        st.session_state.tags = fs.get_tags()

    if st.session_state.tags:
        selected_tag = st.selectbox("Topic", options=st.session_state.tags)

        if st.button("Generate"):
            post = generate_post(selected_length, selected_language, selected_tag,  st.session_state.influencer_id)
            st.write(post)

if __name__ == "__main__":
    main()
