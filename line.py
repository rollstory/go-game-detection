#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
from dataclasses import dataclass


@dataclass
class Line:
    def __init__(
            self,
            x_start: np.int32,
            y_start: np.int32,
            x_end: np.int32,
            y_end: np.int32):

        if not isinstance(x_start, np.int32):
            print('x_start: ', x_start)
            print('type x_start: ', type(x_start))
            raise TypeError('x_start requires an integer value')
        if x_start is None:
            print('x_start: ', x_start)
            raise ValueError('x_start is required')
        if x_start < 0:
            print('x_start: ', x_start)
            raise ValueError('x_start needs to be positive')

        if not isinstance(y_start, np.int32):
            print('y_start: ', y_start)
            print('type y_start: ', type(y_start))
            raise TypeError('y_start requires an integer value')
        if y_start is None:
            print('y_start: ', y_start)
            print('type y_start: ', type(y_start))
            raise ValueError('y_start is required')
        if y_start < 0:
            print('y_start: ', y_start)
            print('type y_start: ', type(y_start))
            raise ValueError('y_start needs to be positive')

        if not isinstance(x_end, np.int32):
            print('type x_end: ', type(x_end))
            print('type x_end: ', type(x_end))
            raise TypeError('x_end requires an integer value')
        if x_end is None:
            print('type x_end: ', type(x_end))
            print('type x_end: ', type(x_end))
            raise ValueError('x_end is required')
        if x_end < 0:
            print('type x_end: ', type(x_end))
            print('type x_end: ', type(x_end))
            raise ValueError('x_end needs to be positive')

        if not isinstance(y_end, np.int32):
            print('type y_end: ', type(y_end))
            print('type y_end: ', type(y_end))
            raise TypeError('y_end requires an integer value')
        if y_end is None:
            print('type y_end: ', type(y_end))
            print('type y_end: ', type(y_end))
            raise ValueError('y_end is required')
        if y_end < 0:
            print('type y_end: ', type(y_end))
            print('type y_end: ', type(y_end))
            raise ValueError('y_end needs to be positive')

        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.line = self.line = np.array([[x_start, y_start, x_end, y_end]])
        self.length_squared = (x_end - x_start)**2 + (y_end - y_start)**2
        if x_end - x_start != 0:
            self.slope = (y_end - y_start) / (x_end - x_start)
        else:
            self.slope = np.inf

    channel = 2

    # Default Hough Lines Parameters
    hough_lines_params = {
        'rho': 1,
        'theta': np.pi/180,
        'threshold': 10,
        'minLineLength': 10,
        'maxLineGap': 10
    }

    @classmethod
    def hough_lines_p(cls, edges: np.ndarray, hough_lines_params=None):
        """
        Apply Hough Line Transform to edge-detected image.

        Parameters
        ----------
            edges (np.ndarray): Edge-detected image.
            hough_lines_params (dict, optional): Parameters for Hough Lines.
            Defaults to None.

        Returns
        -------
            list: List of Line instances representing detected lines.
        """
        if edges is None:
            raise ValueError('edges parameter is None. Use cv.CannyEdge or Frame.adaptive_thresh')

        if hough_lines_params is None:
            hough_lines_params = cls.hough_lines_params
        elif not isinstance(hough_lines_params, dict):
            raise TypeError(
                """
                hough_lines_params must be a dictionary with keys rho, theta,
                 threshold, minLineLength, maxLineGap
                """)

        try:
            lines = cv.HoughLinesP(
                edges,
                rho=hough_lines_params.get('rho'),
                theta=hough_lines_params.get('theta'),
                threshold=hough_lines_params.get('threshold'),
                minLineLength=hough_lines_params.get('minLineLength'),
                maxLineGap=hough_lines_params.get('maxLineGap')
            )
        except cv.error as e:
            print("OpenCV error:", e)
            print("Falling back to standard HoughLinesP parameters")
            hough_lines_params = cls.hough_lines_params
            lines = cv.HoughLinesP(
                edges,
                rho=hough_lines_params.get('rho'),
                theta=hough_lines_params.get('theta'),
                threshold=hough_lines_params.get('threshold'),
                minLineLength=hough_lines_params.get('minLineLength'),
                maxLineGap=hough_lines_params.get('maxLineGap')
            )

        if lines is None:
            return None
        lines = [
            cls(
                line[0][0],
                line[0][1],
                line[0][2],
                line[0][3]) for line in lines]
        return lines

    vertical_lines = []
    horizontal_lines = []

    def filter_line(
            self,
            min_line_length: int,
            horizontal_slope: float,
            vertical_slope: float):
        """
        Filter lines based on length and slope.

        Parameters
        ----------
        min_line_length : int
            Minimum line length to consider.
        horizontal_slope : float
            Maximum absolute slope for horizontal lines.
        vertical_slope : float
            Minimum absolute slope for vertical lines.
        """
        if self.length_squared > min_line_length**2 and np.abs(self.slope) < horizontal_slope:
            Line.horizontal_lines.append(self)
        if self.length_squared > min_line_length**2 and np.abs(self.slope) > vertical_slope:
            Line.vertical_lines.append(self)

    @staticmethod
    def draw_lines(lines: list, bounds: tuple, image: np.ndarray):
        for line in lines:
            cv.line(
                image,
                (line.x_start, line.y_start),
                (line.x_end, line.y_end),
                (255, 255, 255), 1, cv.LINE_AA)

    def extend_line(self, bounds: tuple):
        """
        Method extends a line to the given boundaries

        Parameters
        ----------
        bounds : tuple
            Boundaries for line extension. You might want to choose the
            image boundaries or some user input or find a bounding
            rectangle.

        Returns
        -------
        TYPE
        instance of Line: extended Line
        """

        if not isinstance(bounds, tuple):
            raise TypeError('tuple of two unsigned integers is required')
        else:
            if not bounds[0] > 0:
                error_msg = f'{bounds[0]} needs to be a positive integer'
                raise ValueError(error_msg)
            if not bounds[1] > 0:
                error_msg = f'{bounds[1]} needs to be a positive integer'
                raise ValueError(error_msg)

        x_start, y_start, x_end, y_end = self.line[0]

        if y_end - y_start != 0 and x_end - x_start != 0:
            x_max = x_start + (bounds[0]-1 - y_start) * (x_end - x_start) / (y_end -y_start)
            y_max = y_start + (bounds[1]-1 - x_start) * (y_end - y_start) / (x_end - x_start)
            x_min = x_start + (0 - y_start) * (x_end - x_start) / (y_end - y_start)
            y_min = y_start + (0 - x_start) * (y_end - y_start) / (x_end - x_start)

            if 0 <= x_max < bounds[1]:
                x_end = x_max
                y_end = bounds[0]-1
                if 0 <= x_min < bounds[1]:
                    x_start = x_min
                    y_start = 0
                elif 0 <= y_min < bounds[0]:
                    x_start = 0
                    y_start = y_min
                elif 0 <= y_max < bounds[0]:
                    x_start = bounds[1]-1
                    y_start = y_max
                return Line(
                    np.int32(np.floor(x_start)),
                    np.int32(np.floor(y_start)),
                    np.int32(np.floor(x_end)),
                    np.int32(np.floor(y_end)))

            if 0 <= y_min < bounds[1]:
                x_start = 0
                y_start = y_min
                if (0 <= x_min < bounds[1]):
                    x_end = x_min
                    y_end = 0
                elif (0 <= y_max < bounds[0]):
                    x_end = bounds[1]-1
                    y_end = y_max
                return Line(
                    np.int32(np.floor(x_start)),
                    np.int32(np.floor(y_start)),
                    np.int32(np.floor(x_end)),
                    np.int32(np.floor(y_end)))

            if 0 <= x_min < bounds[1]-1:
                x_start = x_min
                y_start = 0
                x_end = bounds[1]-1
                y_end = y_max
                return Line(
                    np.int32(np.floor(x_start)),
                    np.int32(np.floor(y_start)),
                    np.int32(np.floor(x_end)),
                    np.int32(np.floor(y_end)))

        if y_end - y_start == 0:
            x_start = 0
            x_end = bounds[1]-1
            return Line(
                np.int32(np.floor(x_start)),
                np.int32(np.floor(y_start)),
                np.int32(np.floor(x_end)),
                np.int32(np.floor(y_end)))

        if x_end - x_start == 0:
            y_start = 0
            y_end = bounds[0]-1
            return Line(
                np.int32(np.floor(x_start)),
                np.int32(np.floor(y_start)),
                np.int32(np.floor(x_end)),
                np.int32(np.floor(y_end)))

    horizontal_lines = []
    vertical_lines = []

    @staticmethod
    def sort_horizontal_lines(horizontal_lines: list):
        return sorted(
            horizontal_lines,
            key=lambda x: x.y_start)

    @staticmethod
    def unique_horizontal_lines(
            sorted_horizontal_lines: list,
            epsilon: int = None) -> list:
        """
        Parameters
        ----------
        sorted_horizontal_lines: (: list)
            Takes a list of line objects of lines with an slope of less than 
            1/2, which are supposed to resemble the horizontal lines of a go
            board
        epsilon : (int, np.uint32)
            Since HughLinesP finds more than one parallel line per line on the
            board, we need to unify these lines. Hence we need an error value
            epsilon to create an average line.

        Returns
        -------
        TYPE
            This method returns a list horizontal unique lines, i.e. one line
            for a line on the board.
        """
        if not epsilon:
            epsilon = 5

        same_lines = []
        unique_lines = []
        for i in range(len(sorted_horizontal_lines)):
            x_start, y_start, x_end, y_end = sorted_horizontal_lines[i].line[0]
            if not same_lines:
                same_lines.append(sorted_horizontal_lines[i].line[0])
            else:
                y_start_new = sum([line[1] for line in same_lines]) / len(same_lines)
                y_end_new = sum([line[3] for line in same_lines]) / len(same_lines)

                if np.abs(y_start_new - y_start) <= epsilon and np.abs(y_end_new - y_end) <= epsilon:
                    same_lines.append(sorted_horizontal_lines[i].line[0])
                else:
                    new_line = Line(
                        same_lines[-1][0],
                        np.int32(np.round(y_start_new)),
                        same_lines[-1][2],
                        np.int32(np.round(y_end_new))
                        )
                    unique_lines.append(new_line)
                    same_lines.clear()
                    same_lines.append(sorted_horizontal_lines[i].line[0])
        if same_lines:
            y_start_new = sum([line[1] for line in same_lines]) / len(same_lines)
            y_end_new = sum([line[3] for line in same_lines]) / len(same_lines)
            new_line = Line(
                same_lines[-1][0],
                np.int32(np.round(y_start_new)),
                same_lines[-1][2],
                np.int32(np.round(y_end_new))
            )
            unique_lines.append(new_line)
        return unique_lines

    @staticmethod
    def sort_vertical_lines(vertical_lines: list):
        return sorted(
            vertical_lines,
            key=lambda x: x.x_start)

    @staticmethod
    def unique_vertical_lines(
            sorted_vertical_lines: list,
            epsilon: int = None) -> list:
        """
        Parameters
        ----------
        sorted_vertical_lines: (np.ndarray)
            Takes lines of lines with an slope of less than 1/2, which are
            suppose to resemble the vertical lines of a go board
        epsilon : (int, np.uint32)
            Since HughLinesP finds more than one parallel line per line on the
            board, we need to unify these lines. Hence we need an error value
            epsilon to create an average line.

        Returns
        -------
        TYPE
            This method returns a list vertical unique lines, i.e. one line
            for a line on the board.
        """
        if not epsilon:
            epsilon = 5

        same_lines = []
        unique_lines = []
        for i in range(len(sorted_vertical_lines)):
            x_start, y_start, x_end, y_end = sorted_vertical_lines[i].line[0]
            if not same_lines:
                same_lines.append(sorted_vertical_lines[i].line[0])
            else:
                x_start_new = sum([line[0] for line in same_lines]) / len(same_lines)
                x_end_new = sum([line[2] for line in same_lines]) / len(same_lines)

                if np.abs(x_start_new - x_start) <= epsilon and np.abs(x_end_new - x_end) <= epsilon:
                    same_lines.append(sorted_vertical_lines[i].line[0])
                else:
                    new_line = Line(
                        np.int32(np.round(x_start_new)),
                        same_lines[-1][1],
                        np.int32(np.round(x_end_new)),
                        same_lines[-1][3])
                    unique_lines.append(new_line)
                    same_lines.clear()
                    same_lines.append(sorted_vertical_lines[i].line[0])
        if same_lines:
            x_start_new = sum([line[0] for line in same_lines]) / len(same_lines)
            x_end_new = sum([line[2] for line in same_lines]) / len(same_lines)
            new_line = Line(
                same_lines[-1][0],
                np.int32(np.round(x_start_new)),
                same_lines[-1][2],
                np.int32(np.round(x_end_new))
            )
            unique_lines.append(new_line)
        return unique_lines

    @staticmethod
    def line_intersections(lines: list) -> (list, list):
        """
        Parameters
        ----------
        lines : list
            Pass the unique lines to the method.

        Returns
        -------
        (list, list)
            Intesections of the passed lines.
        """

        intersections = []
        exterior_intersections = []

        for i, line1 in enumerate(lines):
            x1_start, y1_start, x1_end, y1_end = line1.line[0]
            for j, line2 in enumerate(lines[i+1:], start=i+1):
                x2_start, y2_start, x2_end, y2_end = line2.line[0]
                det = (x1_end - x1_start)*(y2_start - y2_end) - (x2_start - x2_end)*(y1_end - y1_start)
                if det != 0:
                    t = ((y2_start - y2_end)*(x2_start - x1_start) + (x2_end - x2_start)*(y2_start - y1_start)) / det
                    s = ((y1_start - y1_end)*(x2_start - x1_start) + (x1_end - x1_start)*(y2_start - y1_start)) / det
                    if 0 <= t <= 1 and 0 <= s <= 1:
                        x_intersect = np.int32(np.round(x1_start + t*(x1_end - x1_start)))
                        y_intersect = np.int32(np.round(y1_start + t*(y1_end - y1_start)))
                        intersections.append([y_intersect, x_intersect])
                    else:
                        x_intersect = np.int32(np.round(x1_start + t*(x1_end - x1_start)))
                        y_intersect = np.int32(np.round(y1_start + t*(y1_end - y1_start)))
                        exterior_intersections.append([y_intersect, x_intersect])
        return (intersections, exterior_intersections)

    @staticmethod
    def draw_intersections(intersections: list, image: np.ndarray):
        """
        Parameters
        ----------
        intersections : list
            intersections of lines.
        image : np.array
            image to draw circle where the intersections are.
        Returns
        -------
        None.
        """
        for center in intersections:
            cv.circle(image, (center[1], center[0]), 5, (255, 255, 0), -1)

    def bresenham_lines(self) -> list:
        """
        Bresenham algorithm to find all points of a line between given
        start and end

        Returns
        -------
        TYPE
            Returns a list with all points of a line.
        """
        x_start, y_start, x_end, y_end = self.line[0]
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
        return points

    outer_pts = []
    bounding_lines = []

    @staticmethod
    def outer_points(channel):

        def click_event(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                if frame[y][x][0] == 255 and frame[y][x][1] == 255 and frame[y][x][2] == 0:
                    dist = {int((pt[0] - x) ** 2 + (pt[1] - y) ** 2): pt for pt in Line.outer_pts}
                    min_dist = min(dist.keys())
                    index = Line.outer_pts.index(dist[min_dist])
                    Line.outer_pts.pop(index)
                else:
                    Line.outer_pts.append([np.int32(x), np.int32(y)])
                if len(Line.outer_pts) == 4:
                    Line.bounding_lines.clear()
                    for i in range(1, len(Line.outer_pts)):
                        Line.bounding_lines.append(
                            Line(
                                Line.outer_pts[i-1][0],
                                Line.outer_pts[i-1][1],
                                Line.outer_pts[i][0],
                                Line.outer_pts[i][1]))
                    Line.bounding_lines.append(
                        Line(
                            Line.outer_pts[0][0],
                            Line.outer_pts[0][1],
                            Line.outer_pts[-1][0],
                            Line.outer_pts[-1][1]))
                else:
                    Line.bounding_lines.clear()

        cv.namedWindow('frame')
        cv.setMouseCallback('frame', click_event)

        cap = cv.VideoCapture(channel)
        if not cap.isOpened():
            print("Cannot open camera")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            try:
                for pt in Line.outer_pts:
                    cv.circle(frame, pt, 5, (255, 255, 0), -1)
                for line in Line.bounding_lines:
                    cv.line(
                        frame,
                        (line.line[0][0], line.line[0][1]),
                        (line.line[0][2], line.line[0][3]),
                        (255, 255, 0))

                cv.imshow('frame', frame)

            except Exception as e:
                print(e)
                break

            if cv.waitKey(1) == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

    def is_outer_line(self) -> bool:

        for line in Line.bounding_lines:
            if Line.line_intersections([line, self])[0]:
                return True
        return False

    collected_intersections = []
