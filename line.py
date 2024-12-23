#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Import the file/class into GoBoard class. Needs maintainance."""

import cv2 as cv
import numpy as np
from dataclasses import dataclass


@dataclass
class Line:
    """Class to deal with line detection and processing."""

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
        self.line = np.array([[x_start, y_start, x_end, y_end]])
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
            raise ValueError('edges parameter is None. Use cv.CannyEdge or \
                             Frame.adaptive_thresh')

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

    @staticmethod
    def tune_hough_lines_params(edges: np.ndarray) -> dict:
        """
        Open an interactive window to adjust the parameters for HoughLinesP.

        Returns the user-defined parameters as a dictionary.

        Parameter:
        ----------
            edges (np.ndarray): Edge-detected image to visualize the results.


        Return:
        ------
            dict: Tuned parameters for HoughLinesP.
        """
        def nothing(x):
            pass

        # Create a window
        cv.namedWindow('Tuning Hough Lines', cv.WINDOW_NORMAL)

        # Initialize trackbars
        cv.createTrackbar(
            'Rho', 'Tuning Hough Lines', 1, 10, nothing)
        cv.createTrackbar(
            'Theta (degrees)', 'Tuning Hough Lines', 1, 180, nothing)
        cv.createTrackbar(
            'Threshold', 'Tuning Hough Lines', 10, 100, nothing)
        cv.createTrackbar(
            'Min Line Length', 'Tuning Hough Lines', 10, 200, nothing)
        cv.createTrackbar(
            'Max Line Gap', 'Tuning Hough Lines', 10, 200, nothing)

        while True:
            # Get current trackbar positions
            rho = cv.getTrackbarPos(
                'Rho', 'Tuning Hough Lines')
            theta = cv.getTrackbarPos(
                'Theta (degrees)', 'Tuning Hough Lines')
            threshold = cv.getTrackbarPos(
                'Threshold', 'Tuning Hough Lines')
            min_line_length = cv.getTrackbarPos(
                'Min Line Length', 'Tuning Hough Lines')
            max_line_gap = cv.getTrackbarPos(
                'Max Line Gap', 'Tuning Hough Lines')

            # Convert theta to radians
            theta_radians = theta * (np.pi / 180)

            # Define current parameters
            params = {
                'rho': rho if rho > 0 else 1,
                'theta': theta_radians if theta > 0 else np.pi / 180,
                'threshold': max(1, threshold),
                'minLineLength': min_line_length,
                'maxLineGap': max_line_gap
            }

            # Apply HoughLinesP with current parameters
            lines = cv.HoughLinesP(
                edges,
                rho=params['rho'],
                theta=params['theta'],
                threshold=params['threshold'],
                minLineLength=params['minLineLength'],
                maxLineGap=params['maxLineGap']
            )

            # Draw detected lines on a blank canvas
            lines_img = np.zeros_like(edges, dtype=np.uint8)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv.line(lines_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Show the result
            cv.imshow('Tuning Hough Lines', lines_img)

            # Break the loop on 'ESC' key
            key = cv.waitKey(10) & 0xFF
            if key == 27:  # ESC key
                break

        cv.destroyAllWindows()
        return params

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
        if self.length_squared > min_line_length**2 \
                and np.abs(self.slope) < horizontal_slope:
            Line.horizontal_lines.append(self)
        if self.length_squared > min_line_length**2 \
                and np.abs(self.slope) > vertical_slope:
            Line.vertical_lines.append(self)

    @classmethod
    def process_horizontal_lines(
            cls,
            lines: list,
            params: dict,
            bounds: tuple) -> list:
        """
        Process horizontal lines in order to intersect with vertical lines.

        Parameters
        ----------
        lines : list
            List of Line objects, generated by HughLines.
        params : dict
            filtering parameters.
        bounds : tuple
            bounds to which lines can be extended.

        Returns
        -------
        list
            Unified and sorted horizontal lines.

        """
        for line in lines:
            cls.filter_line(
                line,
                params.get('min_line_length'),
                params.get('horizontal_slope'),
                params.get('vertical_slope')
                )

        extended_horizontal_lines = [line.extend_line(bounds)
                                     for line in Line.horizontal_lines]

        sorted_horizontal_lines = cls.sort_horizontal_lines(
            extended_horizontal_lines)

        unique_horizontal_lines = cls.unique_horizontal_lines(
            sorted_horizontal_lines, params.get('epsilon_unify'))

        unique_horizontal_lines = [
            line
            for line in unique_horizontal_lines
            if not line.is_outer_line()]

        return unique_horizontal_lines

    @classmethod
    def tune_line_filter_params(
            cls,
            lines: list,
            params: dict,
            frame: np.ndarray
            ) -> dict:
        """
        Debugging method for filtering lines. Might be erroneous.

        Parameters
        ----------
        lines : list
            List of Line objects.
        params : dict
            parameter dictionary to adjust.
        frame : np.ndarray
            numoy array that contain image data.

        Returns
        -------
        dict
            adjusted parameters.

        """

        def nothing(x):
            pass

        # Create a window
        cv.namedWindow('Tuning Filter Lines', cv.WINDOW_NORMAL)

        # Initialize trackbars
        cv.createTrackbar(
            'Minimal Line Length',
            'Tuning Filter Lines', 20, 500, nothing)
        cv.createTrackbar(
            'Horizontal Slope',
            'Tuning Filter Lines', 0, 10, nothing)
        cv.createTrackbar(
            'Vertical Slope',
            'Tuning Filter Lines', 1, 99, nothing)
        cv.createTrackbar(
            'Unifying Epsilon',
            'Tuning Filter Lines', 1, 15, nothing)

        while True:
            # Get current trackbar values
            min_line_length = cv.getTrackbarPos(
                'Minimal Line Length', 'Tuning Filter Lines')
            horizontal_slope = cv.getTrackbarPos(
                'Horizontal Slope', 'Tuning Filter Lines')
            vertical_slope = cv.getTrackbarPos(
                'Vertical Slope', 'Tuning Filter Lines')
            epsilon_unify = cv.getTrackbarPos(
                'Unifying Epsilon', 'Tuning Filter Lines')

            params['min_line_length'] = min_line_length,
            params['horizontal_slope'] = horizontal_slope,
            params['vertical_slope'] = vertical_slope / 100,
            params['epsilon_unify'] = epsilon_unify,

            frame_copy = frame.copy

            for line in lines:
                cls.filter_line(
                    line,
                    params.get('min_line_length'),
                    params.get('horizontal_slope'),
                    params.get('vertical_slope')
                    )

            extended_horizontal_lines = [line.extend_line(frame.shape)
                                         for line in Line.horizontal_lines]

            sorted_horizontal_lines = cls.sort_horizontal_lines(
                extended_horizontal_lines)

            unique_horizontal_lines = cls.unique_horizontal_lines(
                sorted_horizontal_lines, params.get('epsilon_unify'))

            unique_horizontal_lines = [
                line
                for line in unique_horizontal_lines
                if line.is_outer_line()]

            cls.draw_lines(cls.unique_horizontal_lines, frame_copy)
            # Display the processed frame
            cv.imshow('Tuning Filter Lines', frame_copy)

            cls.horizontal_lines.clear()
            cls.vertical_lines.clear()

            # Break the loop on 'ESC' key
            key = cv.waitKey(10) & 0xFF
            if key == 27:  # ESC key
                break

        cv.destroyAllWindows()
        return params

    @staticmethod
    def draw_lines(lines: list, image: np.ndarray):
        """
        Draw lines to an image.

        Parameters
        ----------
        lines : list
            List of Line objects.
        image : np.ndarray
            numpy array containing image data.

        Returns
        -------
        None.

        """
        for line in lines:
            cv.line(
                image,
                (line.x_start, line.y_start),
                (line.x_end, line.y_end),
                (255, 255, 255), 1, cv.LINE_AA)

    def extend_line(self, bounds: tuple):
        """
        Extend a line to the given boundaries.

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
            x_max = x_start + (bounds[0]-1 - y_start) \
                * (x_end - x_start) / (y_end - y_start)

            y_max = y_start + (bounds[1]-1 - x_start) \
                * (y_end - y_start) / (x_end - x_start)

            x_min = x_start + (0 - y_start) \
                * (x_end - x_start) / (y_end - y_start)

            y_min = y_start + (0 - x_start) \
                * (y_end - y_start) / (x_end - x_start)

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
        """
        Sort horizontal lines.

        Parameters
        ----------
        horizontal_lines: (list)
            Takes lines of lines with an slope of less than 1/2, which are
            suppose to resemble the vertical lines of a go board
        Returns
        -------
        TYPE
            This method returns a list horizontal sorted lines.
        """
        return sorted(
            horizontal_lines,
            key=lambda x: x.y_start)

    @staticmethod
    def unique_horizontal_lines(
            sorted_horizontal_lines: list,
            epsilon: int = None) -> list:
        """
        Select unique lines for each set of lines close to each other.

        Parameter
        ----------
        sorted_horizontal_lines: (: list)
            Takes a list of line objects of lines with an slope of less than
            1/2, which are supposed to resemble the horizontal lines of a go
            board.
        epsilon : (int, np.uint32)
            Since HughLinesP finds more than one parallel line per line on the
            board, we need to unify these lines. Hence we need an error value
            epsilon to create an average line.

        Return:
        ------
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
                y_start_new = sum([line[1] for line in same_lines]) \
                    / len(same_lines)
                y_end_new = sum([line[3] for line in same_lines]) \
                    / len(same_lines)

                if np.abs(y_start_new - y_start) <= epsilon \
                        and np.abs(y_end_new - y_end) <= epsilon:
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
            y_start_new = sum([line[1] for line in same_lines]) \
                / len(same_lines)
            y_end_new = sum([line[3] for line in same_lines]) / len(same_lines)
            new_line = Line(
                same_lines[-1][0],
                np.int32(np.round(y_start_new)),
                same_lines[-1][2],
                np.int32(np.round(y_end_new))
            )
            unique_lines.append(new_line)
        return unique_lines

    @classmethod
    def create_vertical_lines(cls, board_size: int) -> (list, np.ndarray):
        """
        Create vertical lines. Needs Line.bounding_lines to not be empty.

        Parameters
        ----------
        board_size: (np.ndarray)
            Takes lines of lines with an slope of less than 1/2, which are
            suppose to resemble the vertical lines of a go board
        Returns
        -------
        TYPE
            This method returns a list of vertical and the horizontal grid.
        """
        lines = [line.bresenham_lines() for line in Line.bounding_lines]

        hor_grid = []
        for line in lines:
            # use only horizontal lines
            # hence outer_lines needs to be generated that way
            if lines.index(line) % 2 == 0:
                x_grid = np.linspace(
                    line[0][1],
                    line[-1][1],
                    num=board_size,
                    dtype=np.int32)
                line_grid = [pt for pt in line if pt[1] in x_grid]
                hor_grid.append(line_grid)

        hor_grid[1].sort(key=lambda x: x[1])
        hor_grid[0].sort(key=lambda x: x[1])
        hor_grid.sort()

        vertical_lines = [cls(
              x_start=np.int32(hor_grid[0][i][1]),
              y_start=np.int32(hor_grid[0][i][0]),
              x_end=np.int32(hor_grid[1][i][1]),
              y_end=np.int32(hor_grid[1][i][0]))
         for i in range(len(hor_grid[0]))]

        return vertical_lines, hor_grid

    @staticmethod
    def sort_vertical_lines(vertical_lines: list):
        """
        Sort vertical lines.

        Parameters
        ----------
        vertical_lines: (np.ndarray)
            Takes lines of lines with an slope of less than 1/2, which are
            suppose to resemble the vertical lines of a go board
        Returns
        -------
        TYPE
            This method returns a list vertical sorted lines.
        """
        return sorted(
            vertical_lines,
            key=lambda x: x.x_start)

    @staticmethod
    def unique_vertical_lines(
            sorted_vertical_lines: list,
            epsilon: int = None) -> list:
        """
        Unifies lines next to each other.

        Parameters
        ----------
        sorted_vertical_lines: (list)
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
                x_start_new = sum([line[0] for line in same_lines]) \
                    / len(same_lines)
                x_end_new = sum([line[2] for line in same_lines]) \
                    / len(same_lines)

                if np.abs(x_start_new - x_start) <= epsilon \
                        and np.abs(x_end_new - x_end) <= epsilon:
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
            x_start_new = sum([line[0] for line in same_lines])/len(same_lines)
            x_end_new = sum([line[2] for line in same_lines])/len(same_lines)
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
        Caculate intersections of lines of a list.

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
                det = (x1_end - x1_start)*(y2_start - y2_end)
                - (x2_start - x2_end)*(y1_end - y1_start)
                if det != 0:
                    t = ((y2_start - y2_end)*(x2_start - x1_start) +
                         (x2_end - x2_start)*(y2_start - y1_start)) / det
                    s = ((y1_start - y1_end)*(x2_start - x1_start) +
                         (x1_end - x1_start)*(y2_start - y1_start)) / det
                    if 0 <= t <= 1 and 0 <= s <= 1:
                        x_intersect = np.int32(
                            np.round(x1_start + t*(x1_end - x1_start)))
                        y_intersect = np.int32(np.round(
                            y1_start + t*(y1_end - y1_start)))
                        intersections.append([y_intersect, x_intersect])
                    else:
                        x_intersect = np.int32(
                            np.round(x1_start + t*(x1_end - x1_start)))
                        y_intersect = np.int32(
                            np.round(y1_start + t*(y1_end - y1_start)))
                        exterior_intersections.append(
                            [y_intersect, x_intersect])
        return (intersections, exterior_intersections)

    @staticmethod
    def draw_intersections(intersections: list, image: np.ndarray):
        """
        Draw intersections of lines in an Image.

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
        for line in intersections:
            for center in line:
                cv.circle(image, (center[1], center[0]), 5, (125, 125, 0), 2)

    def bresenham_lines(self) -> list:
        """
        Bresenham algorithm to find all points of a line between start and end.

        Parameter
        ----------
        None.

        Returns
        -------
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
                points.append([y, x])
                error -= dy
                if error < 0:
                    y += increment_y
                    error += dx
                x += increment_x
        else:
            error = dy / 2
            while y != y_end:
                points.append([y, x])
                error -= dx
                if error < 0:
                    x += increment_x
                    error += dy
                y += increment_y
        points.append([y_end, x_end])
        return points

    outer_pts = []
    bounding_lines = []

    @staticmethod
    def outer_points(channel):
        """
        Generate bounding lines from user input.

        Parameters
        ----------
        channel : int
            Camera channel.

        Returns
        -------
        None.

        """

        def click_event(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                # clicking on a existing point removes it
                if frame[y][x][0] == 255 and frame[y][x][1] == 255 \
                        and frame[y][x][2] == 0:
                    dist = {int((pt[0] - x) ** 2 + (pt[1] - y) ** 2):
                            pt for pt in Line.outer_pts}
                    min_dist = min(dist.keys())
                    index = Line.outer_pts.index(dist[min_dist])
                    Line.outer_pts.pop(index)
                else:
                    Line.outer_pts.append([np.int32(x), np.int32(y)])
                    print(Line.outer_pts)

                if len(Line.outer_pts) == 4:
                    Line.bounding_lines.clear()
                    # change 240, when one gets the camera values
                    angles = np.arctan(
                        np.array(
                            [(240-pt[1]) / pt[0] for pt in Line.outer_pts]))
                    sorted_indices = np.argsort(angles)
                    print(sorted_indices, sorted_indices)
                    zipped = list(zip(Line.outer_pts, sorted_indices))
                    zipped.sort(key=lambda x: x[1], reverse=True)
                    Line.outer_pts = [item[0] for item in zipped]
                    print(Line.outer_pts)

                    for i in range(len(Line.outer_pts)):
                        Line.bounding_lines.append(
                            Line(
                                Line.outer_pts[i][0],
                                Line.outer_pts[i][1],
                                Line.outer_pts[(i+1) % len(Line.outer_pts)][0],
                                Line.outer_pts[(i+1) % len(Line.outer_pts)][1])
                            )
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
        """
        Check if a Line object is out of bounds.

        Returns
        -------
        bool
            Is True if a line intersects with of of the bounding lines.
            Is False otherwise

        """
        for line in Line.bounding_lines:
            if Line.line_intersections([line, self])[0]:
                return True
        return False

    collected_intersections = []
