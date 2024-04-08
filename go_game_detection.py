#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np

def cannyEdge(frame, ker ,min_val, max_val):
    gray = cv.cvtColor(
        frame,
        cv.COLOR_BGR2GRAY
        )
    blur = cv.GaussianBlur(
        gray,
        (ker,ker),
        cv.BORDER_CONSTANT
        )
    canny = cv.Canny(blur, min_val, max_val)
    return canny

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

def drawCont (contours, size, color):
    blank = np.zeros(size, dtype=np.uint8)
    cv.drawContours(
        blank,
        contours,
        -1,
        color,
        thickness=cv.FILLED
        )
    return blank


def bresenhamLines(x_start, x_end, y_start, y_end):
    points = []
    
    dx = abs(x_end - x_start)
    dy = abs(y_end - y_start)
    
    increment_x = 1 if x_start < x_end else -1
    increment_y = 1 if y_start < y_end else -1
    
    x = x_start
    y = y_start
    
    if dx > dy:
        error = dx / 2
        while x != x_end:
            points.append([x, y])
            error -= dy
            if error < 0:
                y += increment_y
                error += dx
            x += increment_x
    else:
        error = dy / 2
        while y != y_end:
            points.append([x, y])
            error -= dx
            if error < 0:
                x += increment_x
                error += dy
            y += increment_y
            
    points.append([x_end, y_end])
    return np.array(points)

def drawLines(lines, intensity, size, line_length_squared, output_lines_drawn=None):
    lines_pts = []
    if lines is not None:
        for line in lines:
            x_start, y_start, x_end, y_end = line[0][0], line[0][1], line[0][2], line[0][3]
            length = (x_end - x_start)**2 + (y_end-y_start)**2
            if length >= line_length_squared:
                pts = bresenhamLines(x_start, x_end, y_start, y_end)
                lines_pts.append(pts)
    
    if output_lines_drawn is None:
        output_lines_drawn = np.zeros(size)
    
    for line in lines_pts:            
        for pt in line:
            x = pt[0]
            y = pt[1]
            output_lines_drawn[y][x] = intensity 

    return output_lines_drawn

def maxRect(edges):
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    max_rect = []
    for contour in contours:
        perimeter = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.04 * perimeter, True)
        
        if len(approx) == 4:
            x, y, w, h = cv.boundingRect(contour)
            area = float(w)*h
            if area > max_area:
                if len(max_rect)>0:
                    max_rect.pop(0)
                max_rect.append(approx)
    return max_rect

    
def lineCrosses(lines, size, intensity, line_length_squared, postion, crosses=None):
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
                    if (postion == 'outer'):
                        x_intersect = int(np.round(x1_start + t*(x1_end - x1_start)))
                        y_intersect = int(np.round(y1_start + t*(y1_end - y1_start)))
                        crosses[y_intersect, x_intersect] = intensity
                    elif (postion == 'inner'):
                        if 0 <= t <= 1 and 0 <= s <= 1:
                            x_intersect = int(np.round(x1_start + t*(x1_end - x1_start)))
                            y_intersect = int(np.round(y1_start + t*(y1_end - y1_start)))
                            crosses[y_intersect, x_intersect] = intensity

                    
    return crosses

def crossCoords(crosses_threshold, channel):
    cap = cv.VideoCapture(channel)
        
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
        
    while True:
        _, frame = cap.read()
        edges_inverted = adaptThresh(
            frame, area_eps=13, 
            threshold_const=3, 
            mode='inverted')
        edges_inverted = cv.morphologyEx(
            edges_inverted, 
            cv.MORPH_CLOSE,
            np.ones((3,3), np.uint8))
        lines = cv.HoughLinesP(
            edges_inverted, 
            1, np.pi / 180,
            30, 50, 50, 10)
        print(type(lines)) 
        crosses = lineCrosses(
            lines,
            size=frame.shape,
            intensity=255,
            line_length_squared = 10**2,
            postion = 'inner'
            )
        kernel = np.ones((9,9), np.uint8)    
        crosses = cv.morphologyEx(
            crosses, 
            cv.MORPH_CLOSE,
            kernel) 
        gray_scale_crosses = cv.cvtColor(
            crosses, 
            cv.COLOR_BGR2GRAY)
        contours, hierarchy = cv.findContours(
            gray_scale_crosses, 
            cv.RETR_EXTERNAL, 
            cv.CHAIN_APPROX_SIMPLE)
        print('contours', len(contours))
        centroids = []
        for contour in contours:
            M = cv.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
        if len(contours) >= crosses_threshold:
            break
    cap.release()
    return frame, centroids, crosses

def drawCrosses (centroids, crosses, frame):
    for center in centroids:
        cv.circle(frame,center, 5, (255,0,255), -1)
    cv.imshow( 'found crosses', frame)
    cv.imshow('crosses', crosses)

    cv.waitKey(0) 
    cv.destroyAllWindows()


def startWebCam(channel):
    cap = cv.VideoCapture(channel)
    if not cap.isOpened():
        print("Cannot open camera")
    while True:
        ret, frame = cap.read()
        if not ret: 
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv.imshow('frame', frame) 
        if cv.waitKey(1) == ord('q'):
             break

    cap.release()
    cv.destroyAllWindows()
    cap = cv.VideoCapture(channel)
        
        
#startWebCam(2)

frame, centroids, crosses = crossCoords(345, 2)

x_sorted = sorted(centroids, key=lambda x: x[0])
xy_sorted = sorted(x_sorted, key=lambda x: x[1])

y_sorted = sorted(centroids, key=lambda x: x[1])
yx_sorted = sorted(y_sorted, key=lambda x: x[0])

centroids_with_length = sorted(
    [[item, item[0]**2 + item[1]**2 ] for item in centroids], 
    key=lambda x: x[1])

lines = np.array([
    [np.array([yx_sorted[0][0], yx_sorted[0][1], yx_sorted[1][0], yx_sorted[1][1]])],
    [np.array([yx_sorted[-2][0], yx_sorted[-2][1], yx_sorted[-1][0], yx_sorted[-1][1]])],
    [np.array([xy_sorted[0][0], xy_sorted[0][1], xy_sorted[1][0], xy_sorted[1][1]])],
    [np.array([xy_sorted[-2][0], xy_sorted[-2][1], xy_sorted[-1][0], xy_sorted[-1][1]])]
    ])

drawn_lines = drawLines(lines, frame.shape, 255, 4, frame)

drawCrosses(centroids, crosses, frame)

"""def main():


        
if __name__ == '__main__':
	main()"""