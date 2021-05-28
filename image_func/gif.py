# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# transform fig to gif
import imageio
import os
from libtiff import TIFF

class Gif:
    def ToGif(self, image_list, gif_name, duration=0.35):
        frames = []
        for image_ in image_list:
            # tiff = TIFF.open(image_, mode="r")
            # frames.append(tiff)  # imageio.imread(image_)
            frames.append(imageio.imread(image_))
            imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
        return


if __name__ == '__main__':
    image_list = [os.path.join('H:/research/flash_drough/GLC_LandUse', image) for image in
                  os.listdir('H:/research/flash_drough/GLC_LandUse') if image.endswith(".tif")]
    gif = Gif()
    gif.ToGif(image_list=image_list, gif_name='Land_Use_1982_to_2015.gif', duration=0.35)