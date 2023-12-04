import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from PIL import Image
import sys
from converter import convert_jpg_to_png


bodypix_model = load_model(download_model(
    BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))

def get_mask(image_path, bodypix_model, threshold=0.75, inversion=False):
    """ Generates a mask from an image using the specified BodyPix model. The
    mask can either represent the person in the image or the background,
    depending on the inversion parameter.

    Args:
        image_path (str): The path to the image file. Accepts .jpg or .png
        formats.
        bodypix_model: The loaded BodyPix model used for generating the person
        segmentation mask.
        threshold (float, optional): The threshold for the mask generation.
        Defaults to 0.75.
        inversion (bool, optional): Determines the nature of the mask.
                                If False, the mask represents the background.
                                If True, the mask represents the person.
                                Defaults to False.

    Returns:
        tuple: A tuple containing two elements:
               - mask_image_path (str): The file path to the saved mask image.
               - image_path (str): The original image path, potentially
               converted to .png format.

    Raises:
        SystemExit: If the image format is not .jpg or .png. """


    if str(image_path).endswith(".jpg"):
        print("convertion to .png")
        convert_jpg_to_png(image_path, str(image_path + ".png"))
        image_path = str(image_path + ".png")
    elif str(image_path).endswith(".png"):
        pass
    else:
        print("Only .jpg or .png are accepted.")
        sys.exit()

    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_array = tf.keras.preprocessing.image.img_to_array(image)

    result = bodypix_model.predict_single(image_array)
    mask = result.get_mask(threshold=threshold)

    mask_np = mask.numpy().squeeze()

    if inversion == True:
        inverted_mask_np = 1 - mask_np
        mask_image = Image.fromarray((inverted_mask_np * 255).astype('uint8'),
                                     'L')
        alpha_image = Image.fromarray((inverted_mask_np * 255).astype('uint8'),
                                     'L')

    if inversion == False:
        mask_image = Image.fromarray((mask_np * 255).astype('uint8'), 'L')
        alpha_image = Image.fromarray((mask_np * 255).astype('uint8'), 'L')

    mask_image.putalpha(alpha_image)

    mask_image_path = image_path.replace('.png', '_mask.png')
    mask_image.save(mask_image_path)

    return mask_image_path, image_path
