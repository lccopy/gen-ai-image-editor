from PIL import Image

def convert_jpg_to_png(input_file, output_file):
    print(f"Trying to convert {input_file} to {output_file}")
    try:
        image_jpg = Image.open(input_file)
        image_rgba = image_jpg.convert('RGBA')
        image_rgba.save(output_file, 'PNG')
        image_jpg.close()
        image_rgba.close()
        print("Convertion done")
    except Exception as e:
        print(f"Error while converting : {e}")
