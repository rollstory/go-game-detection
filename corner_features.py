#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contain experimantal methods to find corner features.

Created on Thu Dec  5 11:06:16 2024

@author: igor
"""

import cv2 as cv
import numpy as np
import random as rng

from line import Line
from utility import Frame


def bounding_rectangle(channel: int) -> list:
    """
    Find the boundings rectangle of a go board.

    Parameters
    ----------
    channel : int
        Camera channel.

    Returns
    -------
    bounding_lines: list
        Returns the lines bounding the go board

    """
    cap = cv.VideoCapture(channel)
    if not cap.isOpened():
        print("Cannot open camera")

    while True:
        bounding_lines = []
        bounding_squares = []
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        try:
            gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            ret_otsu, otsu_binary = cv.threshold(
                gray_image,
                0,
                255,
                cv.THRESH_BINARY+cv.THRESH_OTSU)

            canny = cv.Canny(otsu_binary, 100, 255)

            # find contours --> img_dilation_2
            board_contours, hierarchy = cv.findContours(
                canny,
                cv.RETR_TREE,
                cv.CHAIN_APPROX_SIMPLE
                )

            for contour in board_contours:
                if 20000 < cv.contourArea(contour) < 2000000:
                    # Approximate the contour to a simpler shape
                    epsilon = 0.02 * cv.arcLength(contour, True)
                    approx = cv.approxPolyDP(contour, epsilon, True)

                # Ensure the approximated contour has 4 points (quadrilateral)
                    if len(approx) == 4:
                        pts = [pt[0] for pt in approx]  # Extract coordinates

                        # Define the points explicitly
                        pt1 = tuple(pts[0])
                        pt2 = tuple(pts[1])
                        pt4 = tuple(pts[2])
                        pt3 = tuple(pts[3])

                        bounding_squares.append([cv.boundingRect(contour)])

                        bounding_lines.append(
                            Line(pt1[0], pt1[1], pt2[0], pt2[1]))
                        bounding_lines.append(
                            Line(pt1[0], pt1[1], pt3[0], pt3[1]))
                        bounding_lines.append(
                            Line(pt2[0], pt2[1], pt4[0], pt4[1]))
                        bounding_lines.append(
                            Line(pt3[0], pt3[1], pt4[0], pt4[1]))

            Line.draw_lines(bounding_lines, frame)

            cv.imshow('bounding ractangle', frame)

            if cv.waitKey(10) == ord('q'):
                cap.release()
                cv.destroyAllWindows()
                return bounding_lines, bounding_squares

        except Exception as e:
            print(e)
            break


channel = 2

bounding_rectangle = bounding_rectangle(channel)

cap = cv.VideoCapture(channel)
if not cap.isOpened():
    print("Cannot open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    try:
        threshold = Frame(frame).enhanced_threshold()
        canny = cv.Canny(threshold, 50, 255)

        contours, hierarchy = cv.findContours(
           canny,
           cv.RETR_EXTERNAL,
           cv.CHAIN_APPROX_SIMPLE)

        # Draw contours
        drawing = np.zeros(
            (canny.shape[0], canny.shape[1], 3), dtype=np.uint8)

        for i in range(len(contours)):
            color = (255, 255, 255)
            cv.drawContours(
                drawing,
                contours,
                i,
                color,
                2,
                cv.LINE_8,
                hierarchy,
                0)

        Line.draw_lines(bounding_rectangle[0][:4], frame)

        cv.imshow('threshold', threshold)
        cv.imshow('frame', frame)
        cv.imshow('edges', canny)
        cv.imshow('contours', drawing)

        if cv.waitKey(10) == ord('q'):
            break

    except Exception as e:
        print(e)
        break

cap.release()
cv.destroyAllWindows()
