import cv2
import numpy as np
import glob
from tqdm import tqdm

img_array = []
for filename in tqdm(sorted(glob.glob('./output/results_3/*'))):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('video_output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
