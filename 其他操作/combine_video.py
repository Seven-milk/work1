# code: utf-8
# author: "Xudong Zheng"
# email: Z786909151@163.com
import os
from moviepy.editor import *

# 定义一个数组
home = "D:/手机/"
list_video = [os.path.join(home, list_) for list_ in os.listdir(home)]
video_result = []
for video_path in list_video:
    video = VideoFileClip(video_path)
    video_result.append(video)
    video.close()

# 拼接视频
video_result_c = concatenate_videoclips(video_result)
# 生成目标视频文件
video_result_c.to_videofile(os.path.join(home, "target.mp4"), fps=24, remove_temp=False)

