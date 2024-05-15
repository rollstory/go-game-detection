#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 16:49:03 2024

@author: igor
"""
import cv2 as cv
import numpy as np



def adaptThresh (frame, area_eps, threshold_const, mode):
    gray = cv.cvtColor(
        frame,
        cv.COLOR_BGR2GRAY
        )

    if mode == 'inverted':
        adaptive_thresh = cv.adaptiveThreshold(
            gray, 
            255, 
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY_INV,
            area_eps,
            threshold_const
            )
    else:
        adaptive_thresh = cv.adaptiveThreshold(
            gray, 
            255, 
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            area_eps,
            threshold_const
            )
    return adaptive_thresh

def filteredLines(lines, line_length, horizontal_slope, vertical_slope):
    long_horizontal_lines = []
    long_vertical_lines = []
    for line in lines:
        x_start, y_start, x_end, y_end = line[0]
        length = np.sqrt((x_start-x_end)**2 +(y_start-y_end)**2)
        slope = (y_start-y_end) / (x_start-x_end)

        if length > line_length and np.abs(slope) < horizontal_slope:
            long_horizontal_lines.append(np.array([np.array(line[0])]))
        if length > line_length and np.abs(slope) > vertical_slope:
            long_vertical_lines.append(np.array([np.array(line[0])]))
    return np.array(long_horizontal_lines), np.array(long_vertical_lines)

def extendLines(line, frame):
    x_start, y_start, x_end, y_end = line[0]

    if y_end - y_start != 0 and x_end - x_start != 0:
        x_max = x_start + (frame.shape[0]-1 - y_start) * (x_end - x_start) / (y_end -y_start)
        y_max = y_start + (frame.shape[1]-1 - x_start) * (y_end -y_start) / (x_end - x_start)
        x_min = x_start + (0 - y_start) * (x_end - x_start) / (y_end - y_start)
        y_min = y_start + (0 - x_start) * (y_end - y_start) / (x_end - x_start)

        if 0 <= x_max < frame.shape[1]:
            x_end = x_max
            y_end = frame.shape[0]-1
            if 0 <= x_min < frame.shape[1]:
                x_start = x_min
                y_start = 0
            elif 0 <= y_min < frame.shape[0]:
                x_start = 0
                y_start = y_min
            elif 0 <= y_max < frame.shape[0]:
                x_start = frame.shape[1]-1
                y_start = y_max
            return np.array([np.array([
                int(np.floor(x_start)), 
                int(np.floor(y_start)), 
                int(np.floor(x_end)), 
                int(np.floor(y_end))
                ], dtype=np.uint32)])
                
        if 0 <= y_min < frame.shape[1]:
            x_start = 0
            y_start = y_min
            if (0 <= x_min < frame.shape[1]):
                x_end = x_min
                y_end = 0
            elif (0 <= y_max < frame.shape[0]):
                x_end = frame.shape[1]-1
                y_end = y_max
            return np.array([np.array([
                int(np.floor(x_start)), 
                int(np.floor(y_start)), 
                int(np.floor(x_end)), 
                int(np.floor(y_end))
                ], dtype=np.uint32)])
        
        if 0 <= x_min < frame.shape[1]-1:
            x_start = x_min
            y_start = 0
            x_end = frame.shape[1]-1
            y_end = y_max
            return np.array([np.array([
                int(np.floor(x_start)), 
                int(np.floor(y_start)), 
                int(np.floor(x_end)), 
                int(np.floor(y_end))
                ], dtype=np.uint32)])        
            
    if y_end - y_start == 0:
        x_start = 0
        x_end = frame.shape[1]-1
        return np.array([np.array([
            int(np.floor(x_start)), 
            int(np.floor(y_start)), 
            int(np.floor(x_end)), 
            int(np.floor(y_end))
            ], dtype=np.uint32)])        
        
    if x_end - x_start == 0:
        y_start = 0
        y_end = frame.shape[0]-1
        return np.array([np.array([
            int(np.floor(x_start)), 
            int(np.floor(y_start)), 
            int(np.floor(x_end)), 
            int(np.floor(y_end))
            ], dtype=np.uint32)])
    
def lineCrosses(lines, size, intensity, line_length_squared, crosses=None):
    crosses_pts = []
    exterior_crosses_pts = []
    if crosses is None:
        crosses = np.zeros(size, dtype=np.uint8)
    for i, line1 in enumerate(lines):
        x1_start, y1_start, x1_end, y1_end = line1[0]
        length1_squared = (x1_end - x1_start)**2 + (y1_end - y1_start)**2
        if length1_squared >= line_length_squared:
            for j, line2 in enumerate(lines[i+1:], start=i+1):
                x2_start, y2_start, x2_end, y2_end = line2[0]
                det = (x1_end - x1_start)*(y2_start - y2_end) - (x2_start - x2_end)*(y1_end - y1_start)
                if det != 0:
                    t = ((y2_start - y2_end)*(x2_start - x1_start) + (x2_end - x2_start)*(y2_start - y1_start)) / det
                    s = ((y1_start - y1_end)*(x2_start - x1_start) + (x1_end - x1_start)*(y2_start - y1_start)) / det
                    if 0 <= t <= 1 and 0 <= s <= 1:
                        x_intersect = int(np.round(x1_start + t*(x1_end - x1_start)))
                        y_intersect = int(np.round(y1_start + t*(y1_end - y1_start)))
                        crosses[y_intersect, x_intersect] = intensity
                        crosses_pts.append([y_intersect, x_intersect])
                    else:
                        x_intersect = int(np.round(x1_start + t*(x1_end - x1_start)))
                        y_intersect = int(np.round(y1_start + t*(y1_end - y1_start)))
                        exterior_crosses_pts.append([y_intersect, x_intersect])         
    return crosses, crosses_pts, exterior_crosses_pts


cap = cv.VideoCapture(2)
if not cap.isOpened():
    print("Cannot open camera")

while True:
    ret, frame = cap.read()    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break    
    
    edges = adaptThresh(
        frame,
        area_eps=21,
        threshold_const=5,
        mode='inverted')
        
    contours, _ = cv.findContours(
        edges,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE)
    
    image = np.zeros(frame.shape)    
    cv.drawContours(image, contours, -1, (255, 255, 255), 1)
    
    lines = cv.HoughLinesP(
        edges, 
        rho = 1, 
        theta = np.pi / 180,
        threshold = 20, 
        minLineLength = 10, 
        maxLineGap = 10)

        
    filtered_horizontal_lines, filtered_vertical_lines = filteredLines(
        lines,
        line_length = 400, 
        horizontal_slope = 0.5, 
        vertical_slope = 3)    
              
    extended_horizontal_lines = np.array([ extendLines (line, frame) for line in filtered_horizontal_lines])
    extended_vertical_lines = np.array([ extendLines (line, frame) for line in filtered_vertical_lines])

    sorted_horizontal_lines = np.array(sorted(extended_horizontal_lines, key=lambda x: x[0][1]))
    sorted_vertical_lines = np.array(sorted(extended_vertical_lines, key=lambda x: x[0][0]))
        
    error = 5
    same_lines = []
    unique_lines = []
    for i in range(len(sorted_horizontal_lines)):
        x_start, y_start, x_end, y_end = sorted_horizontal_lines[i][0]
        if not same_lines:
            same_lines.append(sorted_horizontal_lines[i][0])
        else:
            y_start_new = sum([same_lines[i][1] for i in range(len(same_lines))]) / len(same_lines)
            y_end_new = sum([same_lines[i][3] for i in range(len(same_lines))]) / len(same_lines)
            if np.abs( y_start_new - y_start) <= error and np.abs( y_end_new - y_end) <= error:
                same_lines.append(sorted_horizontal_lines[i][0])
            else:
                unique_lines.append(np.array([[x_start, y_start_new, x_end]]))

    black_mask = np.zeros(frame.shape)
    for line in extended_horizontal_lines:
        x_start, y_start, x_end, y_end = line[0]
        cv.line(black_mask, (x_start, y_start), (x_end, y_end), (255,255,255), 1, cv.LINE_AA)
    for line in extended_vertical_lines:
        x_start, y_start, x_end, y_end = line[0]
        cv.line(black_mask, (x_start, y_start), (x_end, y_end), (255,255,255), 1, cv.LINE_AA)           
        
    cv.imshow('frame', frame)
    cv.imshow('black mask', black_mask)

    
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
     
