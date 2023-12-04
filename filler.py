def edit_image_with_background(client, image_path, mask_path, prompt, n=1,
                               size="1024x1024", response_format="url"):
    """ Fills a masked area in an image using OpenAI's image editing API based
    on a given prompt.

    Parameters:
    client (OpenAIClient): The OpenAI client to use for making the API call.
    image_path (str): The file path to the original image that needs editing.
    mask_path (str): The file path to the mask image. The masked area in this
    image defines where the edits will be applied in the original image.
    prompt (str): The text prompt guiding the AI on how to fill the masked area
    in the image.
    n (int, optional): The number of edited images to generate. Defaults to 1.
    size (str, optional): The size of the generated image(s).
    Defaults to "1024x1024".
    response_format (str, optional): The format of the response, e.g.,
    URL of the edited images. Defaults to "url".

    Returns:
    dict: A dictionary containing the response from the API, typically including
    URLs to the edited images. """

    ret = client.images.edit(
        image=open(image_path, "rb"),
        mask=open(mask_path, "rb"),
        prompt=prompt,
        n=n,
        size=size,
        response_format=response_format,
    )
    return ret
