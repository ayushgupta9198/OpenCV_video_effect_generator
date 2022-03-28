import numpy as np
import cv2
import random
import glob
import math


def verify_alpha_channel(frame):
    try:
        frame.shape[3] # looking for the alpha channel
    except IndexError:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    return frame


def apply_hue_saturation(frame, alpha=3, beta=3):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    s.fill(199)
    v.fill(255)
    hsv_image = cv2.merge([h, s, v])

    out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    frame = verify_alpha_channel(frame)
    out = verify_alpha_channel(out)
    frame = cv2.addWeighted(out, 0.25, frame, 1.0, .23, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame



def apply_color_overlay(frame, intensity=0.5, blue=0, green=218, red=0):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA) 
    frame = verify_alpha_channel(frame)
    frame_h, frame_w, frame_c = frame.shape
    sepia_bgra = (blue, green, red, 1)
    overlay = np.full((frame_h, frame_w, 4), sepia_bgra, dtype='uint8')
    frame = cv2.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def apply_sepia(frame, intensity=0.5):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA) 
    frame = verify_alpha_channel(frame)
    frame_h, frame_w, frame_c = frame.shape
    sepia_bgra = (20, 66, 112, 1)
    overlay = np.full((frame_h, frame_w, 4), sepia_bgra, dtype='uint8')
    frame = cv2.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def alpha_blend(frame_1, frame_2, mask):
    alpha = mask/255.0 
    blended = cv2.convertScaleAbs(frame_1*(1-alpha) + frame_2*alpha)
    return blended


def apply_circle_focus_blur(frame, intensity=0.2):
    frame = verify_alpha_channel(frame)
    frame_h, frame_w, frame_c = frame.shape
    y = int(frame_h/2)
    x = int(frame_w/2)

    mask = np.zeros((frame_h, frame_w, 4), dtype='uint8')
    cv2.circle(mask, (x, y), int(y/2), (255,255,255), -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (21,21),11 )

    blured = cv2.GaussianBlur(frame, (21,21), 11)
    blended = alpha_blend(frame, blured, 255-mask)
    frame = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)
    return frame


def portrait_mode(frame):
    # cv2.imshow('frame', frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 120,255,cv2.THRESH_BINARY)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
    blured = cv2.GaussianBlur(frame, (21,21), 11)
    blended = alpha_blend(frame, blured, mask)
    frame = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)
    return frame


def apply_invert(frame):
    return cv2.bitwise_not(frame)


    
def apply_goost(frame, last_frame, count):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA) 
    last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2BGRA) 
    if frame.shape == last_frame.shape:
        frame = cv2.addWeighted(src1=frame, alpha=0.5, src2=last_frame, beta=0.5, gamma=0.0)
    count += 1
    if count == 10:
        last_frame = frame
        count = 0
        
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame, last_frame, count

def apply_mirror(frame):
    frame_h, frame_w, frame_c = frame.shape
    mirror_left = frame[:,:int(frame_w/2),:]    
    mirror_right = cv2.flip(mirror_left, 1)
    mirror = np.concatenate((mirror_left, mirror_right), axis=1)
    return mirror

def apply_vmirror(frame):
    frame_h, frame_w, frame_c = frame.shape
    mirror_left = frame[:int(frame_h/2),:]    
    mirror_right = cv2.flip(mirror_left, 0)
    mirror = np.concatenate((mirror_left, mirror_right), axis=0)
    return mirror


def apply_corners(frame):
    frame_h, frame_w, frame_c = frame.shape
    top_left = frame[:int(frame_h/2), :int(frame_w/2), :]
    top_right = frame[:int(frame_h/2), int(frame_w/2):, :]
    bottom_left = frame[int(frame_h/2):, :int(frame_w/2), :]
    bottom_right = frame[int(frame_h/2):, int(frame_w/2):, :]
    img_half1 = np.concatenate((bottom_right, top_right), axis=1)
    img_half2 = np.concatenate((bottom_left, top_left), axis=1)
    join_corner = np.concatenate((img_half1, img_half2), axis=0)
    return join_corner

def apply_pixelated(frame, w=64, h=64):
    height, width = frame.shape[:2]
    temp = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    pixelated_img = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixelated_img

def apply_hstrip(frame):
    frame_h, frame_w, frame_c = frame.shape
    strip1 = frame[:int(frame_h*0.25), :, :]
    strip2 = frame[int(frame_h*0.25):int(frame_h*0.50),:, :]
    strip3 = frame[int(frame_h*0.50):int(frame_h*0.75), :, :]
    strip4 = frame[int(frame_h*0.75):, :, :]
    strip2 =cv2.flip(strip2, 1)
    strip4 =cv2.flip(strip4, 1)
    img_half1 = np.concatenate((strip1, strip2), axis=0)
    img_half2 = np.concatenate((strip3, strip4), axis=0)
    hstrip_img = np.concatenate((img_half1, img_half2), axis=0)
    return hstrip_img

def apply_vstrip(frame):
    frame_h, frame_w, frame_c = frame.shape
    strip1 = frame[:,:int(frame_w*0.25),:]
    strip2 = frame[:,int(frame_w*0.25):int(frame_w*0.50),:]
    strip3 = frame[:,int(frame_w*0.50):int(frame_w*0.75), :]
    strip4 = frame[:,int(frame_w*0.75):, :]
    strip2 =cv2.flip(strip2, 0)
    strip4 =cv2.flip(strip4, 0)
    img_half1 = np.concatenate((strip1, strip2), axis=1)
    img_half2 = np.concatenate((strip3, strip4), axis=1)
    vstrip_img = np.concatenate((img_half1, img_half2), axis=1)
    return vstrip_img
