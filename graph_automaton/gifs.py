from PIL import Image
import glob
import os

def make_gif(input_dir, output_dir, output_filename="output.gif"):
    frames = [Image.open(image) for image in sorted(glob.glob(os.path.join(input_dir, "*.png")))]
    frames[0].save(
        os.path.join(output_dir, output_filename),
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=100, 
        loop=0 
    )