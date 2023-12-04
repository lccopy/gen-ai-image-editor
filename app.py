import streamlit as st
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from openai import OpenAI
from mask import get_mask
from filler import edit_image_with_background

bodypix_model = load_model(download_model(
    BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))

def main():
    st.title("Image Filler with Generative AI")

    api_key = st.text_input("Enter your OpenAI API key:")

    if api_key:
        client = OpenAI(api_key=api_key)

    else:
        st.warning("Please enter your OpenAI API key.")
        return

    image_file = st.file_uploader("Upload an image", type=["png", "jpg"])

    inversion_options = ["Replace Background", "Replace Person"]
    inversion_choice = st.radio("Choose action:", inversion_options)
    inversion = inversion_choice == "Replace Person"

    threshold = st.slider("Mask Threshold (default = 0.75)", min_value=0.0, max_value=1.0, value=0.75)

    prompt = st.text_input("Enter your prompt:")

    if st.button("Edit Image"):
        if image_file is not None and prompt:
            with open("temp_image.png", "wb") as f:
                f.write(image_file.getbuffer())

            try:
                mask, image = get_mask(image_path="temp_image.png",
                                       bodypix_model=bodypix_model,
                                       threshold=threshold,
                                       inversion=inversion)

                edited = edit_image_with_background(client=client,
                                                    image_path=image,
                                                    mask_path=mask,
                                                    prompt=prompt)
                edited_url = edited.data[0].url

                st.image(edited_url, caption='Edited Image')

            except Exception as e:
                st.error(f"Error processing the image: {e}")

        else:
            st.warning("Please upload an image and enter a prompt.")


if __name__ == "__main__":
    main()
