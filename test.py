
import glob, os
import cv2 as cv
from process import processing
import matplotlib.pyplot as plt
import numpy as np
import time


def detect_image(test_images, process):
    white_yellow_images = list(map(process.select_rgb_white_yellow, test_images))
    # process.show_images(white_yellow_images)

    gray_images = list(map(process.convert_gray_scale, white_yellow_images))
    # process.show_images(gray_images)

    # edge_images = list(map(lambda image: process.detect_edges(image), gray_images))
    # process.show_images(edge_images) 

    # roi_images = list(map(process.select_region, white_yellow_images))
    # process.show_images(roi_images)

    roi_images = list(map(process.roi_region, gray_images))
    # process.show_images(roi_images)

    roi_final_images = list(map(process.roi_final_region, roi_images))
    # process.show_images(roi_final_images)
    
    pt_1 = [0, 125]
    pt_2 = [25, 200]
    pt_3 = [630, 75]
    pt_4 = [605, 0]
    cnt = np.array([[pt_1, pt_2, pt_3, pt_4]])
    # rotate_score_start = process.crop(roi_images[0],cnt)
    # print(rotate_score_start)
    # rotate_score_end = process.crop(roi_images[1],cnt)
    # print(rotate_score_end)
    # rotate_score = process.crop(roi_images[1],cnt)
    # print(rotate_score)
    
    label1, label2, score = process.detect_on_image(roi_images[0], roi_images[1], roi_images[2], cnt)
    print('the device is {},score={} this moment is {}'.format(label1, round(score,3), label2))

def save_start_end_image(video_dir, video_name):
    save_path = os.path.join(r'D:\vscode-project\python\video_stream_analysis',video_name.split('.mp4')[0])
    # print(save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cap = cv.VideoCapture(os.path.join(video_dir,video_name))
    counts = cap.get(7)
    print('this video has %d frames'%counts)
    cap.set(cv.CAP_PROP_POS_FRAMES,0)
    ret, image_start = cap.read()
    # cv.imshow('start_image',image_start)
    cv.imwrite(save_path+'/'+'frame_1.jpg', image_start)
    cap.set(cv.CAP_PROP_POS_FRAMES,counts-1)
    ret, image_end = cap.read()
    # cv.imshow('end_image',image_end)
    cv.imwrite(save_path+'/'+'frame_2.jpg', image_end)
    print('finished save two frame images to %s'%save_path)
    return save_path

def detect_video(video_name, save_frame, process):
    name = video_name
    process.detect_on_video(name, save_frame, )

if __name__ == '__main__':
    video_dir = r'E:\datasets\yjsk_test\yjsk_videos1'
    start = time.time()
    image_dir = save_start_end_image(video_dir,'0001_normal.mp4')
    images = [plt.imread(path) for path in glob.glob(image_dir+'/*.jpg')]
    detect_img = plt.imread(r'D:\vscode-project\python\video_stream_analysis\yjsk\0001_normal\0001_normal_825.jpg')
    images.append(detect_img)
    process = processing()
    # process.show_images(images)
    detect_img_result = detect_image(images, process)
    end = time.time()
    print('Time cost: {}'.format(end-start))