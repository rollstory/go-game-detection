#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import os 


def snap_shot_from_video():
    
    channel = 2
    cap = cv.VideoCapture(channel)
    
    output_dir = "images/training_images"
    os.makedirs(output_dir, exist_ok=True)
    
    file_name = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        k=cv.waitKey(10) & 0XFF
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('gray', gray)

        if k == ord('d'):
            coords = input('Enter cordinates\n')
            x1, y1, x2, y2 = map(int, coords.split(','))

            cropped_image = gray[
                min(y1, y2):max(y1, y2),
                min(x1, x2):max(x1, x2)]
    
        elif k == ord('s'):
            try:
                file_name = input('Enter a filename: \n')
                cv.imwrite(f"{output_dir}/{file_name}.png", cropped_image)
                print(f"Saved {file_name}.png to {output_dir}")
            except Exception as e:
                print(e, 'no cropped_image')
        elif k == ord('l'):
            try:
                last_image = cv.imread(f'{output_dir}/{file_name}.png')
                cv.imshow('selection', last_image)
            except Exception as e:
                print(e, 'no file_name')
        elif k == ord('c'):
            cv.destroyWindow("selection")
        elif k == ord('q'):
            break
    
        try:
            cv.rectangle(
                gray,
                (max(x1, x2), min(y1, y2)),
                (min(x1, x2), max(y1, y2)),
                (0, 0, 0), 2)
            cv.imshow('gray', gray)
        except:
            pass
        
    cap.release()
    cv.destroyAllWindows()
    
def main():
    snap_shot_from_video()
    
if __name__ == '__main__':
	main()