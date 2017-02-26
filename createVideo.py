import imageio
import os

images = os.listdir("run/")
images.sort()

with imageio.get_writer('movie.mp4', mode='I', fps=30) as writer:
    for image_path in images:
        image = imageio.imread('run/' + image_path)
        writer.append_data(image)
