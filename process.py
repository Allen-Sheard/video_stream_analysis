from cProfile import label
from hashlib import new
from itertools import count
import os, glob
from re import T
from tkinter import S
from turtle import distance
from types import new_class
from unittest import result
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# img_path = r'D:\vscode-project\python\video_stream_analysis\0001_normal\frame_1.jpg'

# img = cv.imread(img_path)
# cv.imshow('img',img)
# gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# edges = cv.Canny(gray_img,100,200)
# retval, dst = cv.threshold(gray_img, 127, 255, cv.THRESH_BINARY)
# cv.imshow('binary_img', dst)
# cv.imshow('canny_img', edges)
# cv.waitKey(0)


class processing():
    # show some images
    def show_images(self, images, cmap=None):
        # dsiplay images interface layout
        cols = 2
        rows = (len(images)+1)//cols

        plt.figure(figsize=(15,12))
        for i, image in enumerate(images):
            plt.subplot(rows, cols, i+1)
            cmap = 'gray' if len(image.shape)==2 else cmap
            plt.imshow(image, cmap=cmap)
            # plt.xticks([])
            # plt.yticks([])
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)   # pad images area
        plt.show()

    # show image
    def cv_show(self, name, img):
        cv.imshow(name,img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def select_rgb_white_yellow(self, image):
        # lower = np.uint8([100, 100, 100])
        # upper = np.uint8([255, 255, 255])

        # white_mask = cv.inRange(image, lower, upper)
        # self.cv_show('white_mask', white_mask)
        ret , thresh = cv.threshold(image, 120, 255, cv.THRESH_BINARY_INV)
        # masked = cv.bitwise_and(image, image, mask=white_mask)
        # self.cv_show('masked', masked)
        return thresh

    def convert_gray_scale(self, image):
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def detect_edges(self, image, low_threshold=50, high_threshold=150):
        return cv.Canny(image, low_threshold, high_threshold)

    # 填充多边形
    def filter_region(self, image, vertices):
        mask = np.zeros_like(image)+255
        if len(mask.shape)==2:
            cv.fillPoly(mask, vertices, 0)
            # self.cv_show('mask', mask)
        return cv.bitwise_and(image, mask)

    # 获取目标区域
    def select_region(self, image):
        rows, cols = image.shape[:2]
        pt_1 = [cols*0.20, rows*0.65]
        pt_2 = [cols*0.20, rows*0.90]
        pt_3 = [cols*0.70, rows*0.90]
        pt_4 = [cols*0.70, rows*0.65]
        
        vertices = np.array([[pt_1, pt_2, pt_3, pt_4]],dtype=np.int32)
        point_img = image.copy()
        # point_img = cv.cvtColor(point_img, cv.COLOR_GRAY2BGR)
        for point in vertices[0]:
            cv.circle(point_img, (point[0],point[1]), 10, (0,0,255), 4)
        # self.cv_show('point_img', point_img)

        return self.filter_region(image, vertices)

    def roi_region(self, image):
        rows, cols = image.shape[:2]
        roi = image[460:660,250:880]

        # self.cv_show('roi_region', roi)
        return roi

    def roi_final_region(self, image):
        pt_1 = [0, 125]
        pt_2 = [25, 200]
        pt_3 = [630, 75]
        pt_4 = [605, 0]
        vertices = np.array([[pt_1, pt_2, pt_3, pt_4]],dtype=np.int32)
        point_img = image.copy()
        for point in vertices[0]:
            cv.circle(point_img, (point[0],point[1]), 10, (0,0,255), 4)
        # self.cv_show('point_img', point_img)
        return self.filter_region(image, vertices)
        

    def drow_box(self, img, cnt):
        
        rect_box = cv.boundingRect(cnt)
        rotated_box = cv.minAreaRect(cnt)

        cv.rectangle(img, (rect_box[0], rect_box[1]), (rect_box[0] + rect_box[2], rect_box[1] + rect_box[3]), (0, 255, 0), 2)

        box = cv.boxPoints(rotated_box)
        box = np.int0(box)
        cv.drawContours(img, [box], 0, (0, 0, 255), 2)
        cv.imwrite('contours.png', img)
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # plt.imshow(img)
        # plt.show()

        return img, rotated_box, box

    def crop(self, img, cnt):
        horizon = True

        img, rotated_box, _ = self.drow_box(img, cnt)

        center, size, angle = rotated_box[0], rotated_box[1], rotated_box[2]
        center, size = tuple(map(int, center)), tuple(map(int, size))

        # print(angle)

        if horizon:
            if size[0] < size[1]:
                angle -= 270
                w = size[1]
                h = size[0]
            else:
                w = size[0]
                h = size[1]
            size = (w, h)

        height, width = img.shape[0], img.shape[1]

        M = cv.getRotationMatrix2D(center, angle, 1)
        img_rot = cv.warpAffine(img, M, (width, height))
        img_crop = cv.getRectSubPix(img_rot, size, center)

        # self.cv_show('img', img)
        # self.cv_show('img_rot', img_rot)
        # self.cv_show('img_crop',img_crop)
        cv.imwrite('rotate_image.png', img_crop)
        return self.score_compute(img_crop)

    # 计算所需区域面积占总面积的比例
    def score_compute(self, image):
        sum = image.shape[0]*image.shape[1]
        # print(sum)
        cnt_array = np.where(image,0,1)
        score = 1-(np.sum(cnt_array)/sum)
        return score

    def detect_on_image(self, start_image, end_image, detect_image, cnt):
        start_score = self.crop(start_image, cnt)   #初始帧的面积得分
        end_score = self.crop(end_image, cnt)       #尾帧的面积得分
        score = self.crop(detect_image, cnt)        #待检测帧的面积得分

        #刀闸整体动作状态、某时刻刀闸的分合状态
        state_lable = ['closed-to-open','opened-to-close']
        detect_label = ['closed','opened unormal','opened normal', 'opened',
                        'closed unormal','closed normal']

        #根据初始和结尾得分判断刀闸整体的动作状态，然后根据某时刻的得分判断当前帧的状态
        #由两个状态综合判断视频中刀闸分合动作是否完成
        if start_score > end_score:
            # print('the device is closed to open')
            label1 = state_lable[0]
            if start_score-0.05 <= score <= start_score+0.05:
                label2 = detect_label[0]
                # print('score=%f this moment is closed'%score)
            elif end_score < score < start_score:
                label2 = detect_label[1]
                # print('score=%f this moment is opened unormal'%score)
            elif end_score-0.05 <= score <= end_score:
                label2 = detect_label[2]
                # print('score=%f this moment is opened normal'%score)
        else: 
            # print('the device is opened to close')
            label1 = state_lable[1]
            if start_score-0.05 <= score <= start_score+0.05:
                label2 = detect_label[3]
                # print('score=%f this moment is opened'%score)
            elif start_score < score < end_score:
                label2 = detect_label[4]
                # print('score=%f this moment is closed unormal'%score)
            elif end_score-0.05 <= score <= end_score:
                label2 = detect_label[5]
                # print('score=%f this moment is closed normal'%score)
        return label1, label2, score

    def detect_on_video(self, video_name, save_frames, cnt, ret=True):
        cap = cv.VideoCapture(video_name)
        count = 0
        while ret:
            ret, image = cap.read()
            count += 1
            if count == 25:
                count = 0

            # new_image = np.copy(image)
            new_image = plt.imread(image)
            white_black_image = self.select_rgb_white_yellow(new_image)
            gray_image = self.convert_gray_scale(white_black_image)
            roi_image = self.roi_region(gray_image)
            label_state, label_detect, score = self.detect_on_image(save_frames, roi_image, cnt)

            text = '{}\n{} {}'.format(label_state, score, label_detect)
            cv.putText(image, text, (100, 150), cv.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,0,255), 2)
            cv.imshow('video',image)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
            
        cv.destroyAllWindows()
        cap.release()

    # 霍夫直线检测
    def hough_lines(self, image):
        return cv.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=30, minLineLength=300, maxLineGap=5)
    
    def draw_lines(self, image, lines, color=[255,0,0], thickness=2, make_copy=True):
        if make_copy:
            image = np.copy(image)
        cleaned = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                x1 = line[0][0]
                y1 = line[0][1]
                x2 = line[0][2]
                y2 = line[0][3]
                cleaned.append(x1,y1,x2,y2)
                cv.line(image, (x1,y1), (x2,y2), color, thickness)
        print('No lines detected:', len(cleaned))
        return image

    def identify_blocks(self, image, lines, make_copy=True):
        if make_copy:
            new_image = np.copy(image)

        cleaned = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                if abs(y2-y1) <=1 and abs(x2-x1) >=25 and abs(x2-x1) <= 55:
                    cleaned.append(x1,y1,x2,y2)

        import operator
        list1 = sorted(cleaned, key=operator.itemgetter(0,1))

        clusters = {}
        dIndex = 0
        clus_dist = 10

        for i in range(len(list1)-1):
            distance = abs(list1[i+1][0]-list1[i][0])
            if distance <= clus_dist:
                if not dIndex in clusters.keys(): clusters[dIndex]=[]
                clusters[dIndex].append(list1[i])
                clusters[dIndex].append(list1[i+1])
            else:
                dIndex += 1

        rects = {}
        i = 0
        for key in clusters:
            all_list = clusters[key]
            cleaned = list(set(all_list))
            if len(cleaned) > 5:
                cleaned = sorted(cleaned, key=lambda tup: tup[1])
                avg_y1 = cleaned[0][1]
                avg_y2 = cleaned[-1][1]
                avg_x1 = 0
                avg_x2 = 0
                for tup in cleaned:
                    avg_x1 += tup[0]
                    avg_x2 += tup[2]
                avg_x1 = avg_x1/len(tup)
                avg_x2 = avg_x2/len(tup)
                rects[i] = (avg_x1, avg_y1, avg_x2, avg_y2)
                i += 1

        print('Num processing lanes:', len(rects))

        buff = 7
        for key in rects:
            tup_topleft = (int(rects[key][0] - buff), int(rects[key][1]))
            tup_botright = (int(rects[key][2] + buff), int(rects[key][3]))
            cv.rectangle(new_image, tup_topleft, tup_botright, (0,255,0), 3)
        return new_image, rects

    def save_images_for_cnn(self, image, spot_dict, folder_name ='cnn_data'):
        for spot in spot_dict.keys():
            (x1, y1, x2, y2) = spot
            (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
            #裁剪
            spot_img = image[y1:y2, x1:x2]
            spot_img = cv.resize(spot_img, (0,0), fx=2.0, fy=2.0) 
            spot_id = spot_dict[spot]
            
            filename = 'spot' + str(spot_id) +'.jpg'
            print(spot_img.shape, filename, (x1,x2,y1,y2))
            
            cv.imwrite(os.path.join(folder_name, filename), spot_img)

   

'''
# 使用sift算子获得特征点，再用FLANN最近邻近似匹配画出匹配点之间的连线
img2 = cv.imread(r'video_stream_analysis\yjsk\0001_normal\0001_normal_825.jpg')
img1 = cv.imread(r'video_stream_analysis\0001_normal\box1.png')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# min_hessian = 1000
# sift = cv.SIFT_create(min_hessian)
sift = cv.SIFT_create()
# kp = sift.detect(gray1,None)
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

img3 = cv.drawKeypoints(gray1,kp1,img1)
# cv.imshow('sift_keypoints_0.jpg',img3)
img4 = cv.drawKeypoints(gray2,kp2,img2)
# cv.imshow('sift_keypoint_500.jpg',img4)
img=cv.drawKeypoints(gray1,kp1,img1,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('video_stream_analysis/sift_keypoints2.jpg',img)

# FLANN参数
index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)
# 寻找最近邻近似匹配
flann = cv.FlannBasedMatcher(index_params, search_params)
# 使用knnMatch匹配处理，返回匹配matches
matches = flann.knnMatch(des1, des2, k=2)
# 通过掩码计算有用的点
matchesMask = [[0,0] for i in range(len(matches))]
# 通过coff系数决定匹配的有效关键点数量
coff = 0.7
# 通过描述符的距离选择需要的点
for i, (m,n) in enumerate(matches):
    if m.distance < coff * n.distance:
        matchesMask[i] = [1, 0]
        draw_params = dict(matchColor=(0,255,0),
                           singlePointColor=(255,0,0),
                           matchesMask=matchesMask,
                           flags=2)
# 使用drawMatchesKnn画出匹配点之间的连线
resultimg1 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
resultimg2 = cv.drawMatchesKnn(gray1,kp1,gray2,kp2,matches,None,**draw_params)
# savepath = 'video_stream_analysis/sift'
# os.makedirs(savepath,exist_ok=True)
# cv.imwrite(os.path.join(savepath,'open.jpg'),resultimg1)
# cv.namedWindow('MatchResult1',cv.WINDOW_NORMAL)
# cv.namedWindow('MatchResult2',cv.WINDOW_NORMAL)
# cv.imshow('MatchResult1',resultimg1)
# cv.imshow('MatchResult2',resultimg2)    
# cv.waitKey(0)
'''

'''
角点检测 Fast ORB

img = cv.imread(r'video_stream_analysis\yjsk\0001_normal\0001_normal_0.jpg',0) 
# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()
# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
cv.imshow('fast_true.png', img2)
# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)
print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
cv.imshow('fast_false.png', img3)
cv.waitKey(0)

img = cv.imread('video_stream_analysis/1.png',0)
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img,None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
cv.imshow('orb.png',img2), cv.waitKey(0)
'''

img1 = cv.imread(r'video_stream_analysis\yjsk\0001_normal\0001_normal_0.jpg', 0)
img2 = cv.imread(r'video_stream_analysis\yjsk\0001_normal\0001_normal_825.jpg', 0)
# img1 = cv.rotate(img1, rotateCode=cv.ROTATE_90_CLOCKWISE)

# orb = cv.ORB_create()

# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)
# # BF匹配
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# matchs = bf.match(des1, des2)
# matchs = sorted(matchs, key = lambda x:x.distance)

# img3 = cv.drawMatches(img1,kp1,img2,kp2,matchs[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3)
# plt.show()

# Initiate SIFT detector
# sift = cv.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# # BFMatcher with default params
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des1,des2, k=2)
# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
# # cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()


# # Initiate SIFT detector
# sift = cv.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# # FLANN parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)   # or pass empty dictionary
# flann = cv.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(des1,des2,k=2)
# # Need to draw only good matches, so create a mask
# matchesMask = [[0,0] for i in range(len(matches))]
# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.7*n.distance:
#         matchesMask[i]=[1,0]
# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = cv.DrawMatchesFlags_DEFAULT)
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
# plt.imshow(img3),plt.show()


# MIN_MATCH_COUNT = 10
# # Initiate SIFT detector
# sift = cv.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
# flann = cv.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des1,des2,k=2)
# # store all the good matches as per Lowe's ratio test.
# good = []
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)

# if len(good)>MIN_MATCH_COUNT:
#     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
#     matchesMask = mask.ravel().tolist()
#     h,w = img1.shape
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv.perspectiveTransform(pts,M)
#     img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
# else:
#     print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
#     matchesMask = None

# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)
# img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
# plt.imshow(img3, 'gray'),plt.show()


# img = cv.imread(r'video_stream_analysis\yjsk\0001_normal\0001_normal_825.jpg')
# blur = cv.GaussianBlur(img,(5,5),0)
# edges = cv.Canny(blur,100,200)
# plt.subplot(221)
# plt.imshow(img)
# plt.title('original img'),plt.xticks([]),plt.yticks([])
# plt.subplot(222)
# plt.imshow(blur)
# plt.title('blur img'),plt.xticks([]),plt.yticks([])
# plt.subplot(223),plt.imshow(edges)
# plt.title('canny img'),plt.xticks([]),plt.yticks([])
# plt.show()


'''
 # 霍夫变换和概率霍夫变换

img = cv.imread(r'video_stream_analysis\0001_normal\box.png')
img1 = img.copy()
img2 = img.copy()
img = cv.GaussianBlur(img, (3, 3), 0)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150, apertureSize=3)
lines = cv.HoughLines(edges, 1, np.pi/90, 150)
 
for line in lines:
    rho = line[0][0]    #像素
    theta = line[0][1]  #弧度
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
 
    cv.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 2)
 
lines = cv.HoughLinesP(edges, 1, np.pi/90, 50, 500, 5)
 
for line in lines:
    x1 = line[0][0]
    y1 = line[0][1]
    x2 = line[0][2]
    y2 = line[0][3]
    cv.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
 
cv.imshow('houghlines3', img1)
cv.imwrite('video_stream_analysis/0001_normal/houghlines1.jpg',img1)
cv.imshow('edges', img2)
cv.waitKey(0)
# print(lines)
'''
 
''' 
def nothing(x):  # 滑动条的回调函数
    pass

 
src = cv.imread(r'video_stream_analysis\yjsk\0001_normal\0001_normal_0.jpg')
srcBlur = cv.GaussianBlur(src, (3, 3), 0)
gray = cv.cvtColor(srcBlur, cv.COLOR_BGR2GRAY)
thresh = cv.threshold(gray,20,255,cv.THRESH_BINARY)[1]
edges = cv.Canny(gray, 50, 150, apertureSize=3)
# WindowName = 'HoughLines'  # 窗口名
# cv.namedWindow(WindowName, cv.WINDOW_AUTOSIZE)  # 建立空窗口
 
# cv.createTrackbar('threshold', WindowName, 0, 100, nothing)  # 创建滑动条
 
# while(1):
#     img = src.copy()
#     threshold = 100 + 2 * cv.getTrackbarPos('threshold', WindowName)  # 获取滑动条值
 
#     lines = cv.HoughLines(edges, 1, np.pi/180, threshold)
 
#     for line in lines:
#         rho = line[0][0]
#         theta = line[0][1]
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         x1 = int(x0 + 1000*(-b))
#         y1 = int(y0 + 1000*(a))
#         x2 = int(x0 - 1000*(-b))
#         y2 = int(y0 - 1000*(a))
 
#         cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
 
#     cv.imshow(WindowName, img)
#     k = cv.waitKey(1) & 0xFF
#     if k == 27:
#         break
# cv.destroyAllWindows()

WindowName = 'HoughLinesP'
cv.namedWindow(WindowName, cv.WINDOW_NORMAL)
cv.createTrackbar('threshold', WindowName, 0, 100, nothing)  # 创建滑动条
cv.createTrackbar('minLineLength', WindowName, 0, 1000, nothing)  # 创建滑动条
cv.createTrackbar('maxLineGap', WindowName, 0, 50, nothing)  # 创建滑动条
cv.createTrackbar('k',WindowName, 1,10,nothing)
cv.createTrackbar('c',WindowName,1,8,nothing)
cv.createTrackbar('r',WindowName,1,8,nothing)

while(1):
    img = edges.copy()
    
    threshold = cv.getTrackbarPos('threshold', WindowName)  # 获取滑动条值
    minLineLength = 2 * cv.getTrackbarPos('minLineLength', WindowName)  # 获取滑动条值
    maxLineGap = cv.getTrackbarPos('maxLineGap', WindowName)  # 获取滑动条值
 
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength, maxLineGap)
 
    for line in lines:
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[0][2]
        y2 = line[0][3]
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
 
    k = cv.getTrackbarPos('k',WindowName)
    c = cv.getTrackbarPos('c',WindowName)
    r = cv.getTrackbarPos('r',WindowName)
    kernel = np.ones((k,k),np.uint8)
    g = cv.getStructuringElement(cv.MORPH_RECT,(r,c))
    g1 = cv.getStructuringElement(cv.MORPH_RECT,(c,r))

    thresh = cv.dilate(img,kernel,iterations=2)
    thresh = cv.erode(thresh,kernel,iterations=1)
    thresh = cv.morphologyEx(thresh,cv.MORPH_OPEN,g)
    thresh = cv.morphologyEx(thresh,cv.MORPH_OPEN,g1)
    thresh = cv.morphologyEx(thresh,cv.MORPH_CLOSE,g1)
    thresh = cv.morphologyEx(thresh,cv.MORPH_CLOSE,g)
    cv.imshow(WindowName, thresh)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
'''