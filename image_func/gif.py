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
    home = 'H:/work/daimeng/20210531gif/gif1'
    image_list = [os.path.join(home, image) for image in
                  os.listdir(home) if image.endswith(".png")]
    gif = Gif()
    gif.ToGif(image_list=image_list, gif_name='1.gif', duration=0.35)