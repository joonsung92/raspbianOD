# 패키지 가져오기
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys

# 카메라 해상도 설정
IM_WIDTH = 1040
IM_HEIGHT = 592

# 카메라 종류선택(파이캠 or usb캠) 뒤에 --usbcam 입력시 불러올 스크립트
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

sys.path.append('..')

# 유틸리티 가져오기
from utils import label_map_util
from utils import visualization_utils as vis_util

# 객체감지모듈
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# 현재 작업경로에 가져옴
CWD_PATH = os.getcwd()

# 객체탐지에 사용되는 모델을 포함한 감지그래프 .pb 파일의 경로를 고정시킴
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# 레이블 맵 파일의 경로
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# 객체 탐지기가 식별할 수 있는 클래스의 수
NUM_CLASSES = 80

# 레이블 맵 불러오기
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# 텐서플로우 모델을 메모리에 로드
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# 객체 탐지기의 입력 및 출력 텐서(데이터)에 대한 정의

# 입력텐서: 이미지
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# 출력텐서: 탐지 상자, 점수, 클래스
# 각 상자는 특정 물체가 감지된 이미지의 일부를 나타냄
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# 점수는 객체에 대한 신뢰도를 표시
# 출력이미지에 탐지된 클래스와 함께 신뢰도를 표시
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# 탐지된 개체 수
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# FPS 계산 초기화
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# 카메라 초기화 및 물체탐지 수행 (picam, usbwebcam에 따라 다르게 설정)

# Picamera 일때
if camera_type == 'picamera':
    # Picamera 초기화 및 raw capture 참조
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        
        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_expanded = np.expand_dims(frame, axis=0)

        # 입력된 이미지를 모델을 실행하여 실제 감지 수행
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # 검출결과 영상처리
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,255),2,cv2.LINE_AA)

        # 영상처리 된 프레임을 출력
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # 종료키: q
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()

# USB webcam 일때
elif camera_type == 'usb':

    camera = cv2.VideoCapture(0)
    ret = camera.set(3,IM_WIDTH)
    ret = camera.set(4,IM_HEIGHT)

    while(True):

        t1 = cv2.getTickCount()

        ret, frame = camera.read()
        frame_expanded = np.expand_dims(frame, axis=0)

        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.85)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,255),2,cv2.LINE_AA)
        
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()

cv2.destroyAllWindows()

