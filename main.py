#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Driver Code."""

from line import Line
from transformation import Transformation

import cv2 as cv
import numpy as np


channel = 2

Line.outer_points(channel)
outer_pts = Line.outer_pts
bounding_lines = Line.bounding_lines
vertical_lines, hor_grid = Line.create_vertical_lines(19)

longest_bounding_line = max(bounding_lines, key=lambda x: x.length_squared)

transformation = Transformation()

transform_longest_bounding_line_pts = transformation.transform_lowest_line(
    longest_bounding_line.x_start,
    longest_bounding_line.y_start,
    longest_bounding_line.x_end,
    longest_bounding_line.y_end,
    )

lower_hor_line = Line(
    transform_longest_bounding_line_pts[0],
    transform_longest_bounding_line_pts[1],
    transform_longest_bounding_line_pts[2],
    transform_longest_bounding_line_pts[3])

vertical_shift = np.round(
    np.sqrt(lower_hor_line.length_squared)).astype(np.int32)

lower_hor_line_shift = np.int32((640 - vertical_shift) / 2)

calib_vertical_grid = np.round(
    np.linspace(
        lower_hor_line_shift,
        lower_hor_line_shift + vertical_shift,
        num=19, endpoint=True)).astype(np.int32)

calib_horizontal_lines = [
    Line(lower_hor_line.x_start, y, lower_hor_line.x_end, y)
    for y in calib_vertical_grid]

calib_horizontal_grid = np.round(
    np.linspace(
        lower_hor_line.x_start,
        lower_hor_line.x_end,
        num=19, endpoint=True)).astype(np.int32)

calib_vertical_lines = [
    Line(
        x, calib_horizontal_lines[0].y_start,
        x, calib_horizontal_lines[-1].y_start)
    for x in calib_horizontal_grid]

found_board_lines = [
    *[line for line in bounding_lines if np.abs(line.slope) < 1],
    *vertical_lines
    ]

calib_intersections = np.array(
    [np.array([i, j])
     for j in calib_vertical_grid
     for i in calib_horizontal_grid],
    dtype=np.int32
    )

for line in vertical_lines:
    pass

black = np.zeros((480, 640, 3), dtype=np.uint8)
Line.draw_lines(found_board_lines, black)

black_copy = np.zeros((640, 640, 3), dtype=np.uint8)
Line.draw_lines(calib_horizontal_lines, black_copy)
Line.draw_lines(calib_vertical_lines, black_copy)

cv.imshow('black', black)
cv.imshow('black_copy', black_copy)

cv.waitKey(0)
cv.destroyAllWindows()
