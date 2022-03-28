import cv2
import numpy as np
from effects import *
import os


def func(input_path, output_path, effect):
    # cap = cv2.VideoCapture("/home/saktiman/Dev-ai/video_effects/ice_digonal_top_leftto_bottom_right.mp4")
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(5))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(output_path, fourcc, fps, (width,height)) 
    # Pre-preprocessing should be done here
    _, last_frame = cap.read()
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if effect == apply_goost:
            frame, last_frame, count = effect(frame, last_frame, count)
        else :
            frame = effect(frame)           
        output.write(frame)
    cap.release()
    output.release()
    cv2.destroyAllWindows()

def main():
    print("/***************started*********************/")  
    input_path = "/home/ayush-ai/Music/video_effects/21 April/video_effects/1.mp4"
    input_vid_name = input_path.split('/')[-1].split('.')[0]
    os.makedirs('./output', exist_ok=True)
    
    effect_list = [apply_hue_saturation, apply_color_overlay, apply_sepia, apply_circle_focus_blur,
                   portrait_mode, apply_invert, apply_goost, apply_mirror, apply_vmirror,
                   apply_corners, apply_pixelated, apply_hstrip, apply_vstrip]
    effect_name_list = ['hue_saturation', 'color_overlay', 'sepia', 'circle_focus_blur',
                   'portrait_mode', 'invert', 'goost', 'mirror', 'vmirror',
                   'corners', 'pixelated', 'hstrip', 'vstrip']
    
    for effect_name, effect in zip(effect_name_list, effect_list):
        print(f"executing ...{effect_name}")
        output_path = f"./output/{input_vid_name}_{effect_name}.mp4"
        func(input_path, output_path, effect)
    
    print("/****************ended*********************/")


if __name__=='__main__':
    main()
