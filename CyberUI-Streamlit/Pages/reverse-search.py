import streamlit as st
from google_img_source_search import ReverseImageSearcher, SafeMode
import pandas as pd
import tempfile
import os

# Function to perform reverse image search by URL
def search_by_url(image_url):
    rev_img_searcher = ReverseImageSearcher()
    return rev_img_searcher.search(image_url)

# Function to perform reverse image search by file
def search_by_file(image_file):
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.name)[1]) as temp_file:
        temp_file.write(image_file.read())
        temp_file_path = temp_file.name
    
    rev_img_searcher = ReverseImageSearcher()
    results = rev_img_searcher.search_by_file(temp_file_path)
    
    # Clean up the temporary file after searching
    os.remove(temp_file_path)
    
    return results

# Function to display search results and export to CSV
def display_results(results):
    search_data = []
    if results:
        for search_item in results:
            search_data.append({
                'Title': search_item.page_title,
                'Site': search_item.page_url,
                'Image URL': search_item.image_url
            })
        export_to_csv(search_data)
    else:
        st.write("No results found.")

# Function to export results to CSV
def export_to_csv(data, filename='search_results.csv'):
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name=filename, mime='text/csv')

# Streamlit app
def main():
    st.title("Reverse Image Search")

    # Input for image URL
    image_url = st.text_input("Enter Image URL:")

    # Upload an image
    uploaded_image = st.file_uploader("Or upload an image:", type=["png", "jpg", "jpeg"])

    # Perform search on input
    if st.button("Search"):
        if uploaded_image:
            # Perform reverse image search by file
            results = search_by_file(uploaded_image)
        elif image_url:
            # Perform reverse image search by URL
            results = search_by_url(image_url)
        else:
            st.error("Please provide either an image URL or upload an image.")
            return

        # Display search results
        display_results(results)

if __name__ == "__main__":
    main()
