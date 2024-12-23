#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np


class Frame:
    def __init__(self, frame: np.ndarray):
        if not isinstance(frame, np.ndarray):
            raise TypeError('Accepts OpenCV images only')
        self.frame = frame

    adaptive_thresh_params_reg_alt = {
        'max_value': 255,
        'adaptive_method': cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        'threshold_type': cv.THRESH_BINARY,
        'block_size': 13,
        'threshold_const': 3
        }
    adaptive_thresh_params_reg = {
        'max_value': 255,
        'adaptive_method': cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        'threshold_type': cv.THRESH_BINARY,
        'block_size': 21,
        'threshold_const': 5
        }

    adaptive_thresh_params_alt = {
        'max_value': 255,
        'adaptive_method': cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        'threshold_type': cv.THRESH_BINARY_INV,
        'block_size': 13,
        'threshold_const': 3
        }
    adaptive_thresh_params = {
        'max_value': 255,
        'adaptive_method': cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        'threshold_type': cv.THRESH_BINARY_INV,
        'block_size': 21,
        'threshold_const': 5
        }

    # inverted thresh worked better

    def adaptive_thresh(
            self,
            adaptive_thresh_params: dict = None) -> np.ndarray:
        """
        Parameters
        ----------
        adaptive_thresh_params : dict, optional
            This dictionary contains the parameters in order to apply
            adaptive thresholding. Standard parameter are testet for specific
            setting. The default is None.

        Raises
        ------
        TypeError
            Just in case some one misses this.

        Returns
        -------
        adaptive_thresh : np.ndarray
            Applies opencv's adaptive threshold to a frame.
        """
        if adaptive_thresh_params is None:
            adaptive_thresh_params = Frame.adaptive_thresh_params
        else:
            if not isinstance(adaptive_thresh_params, dict):
                raise TypeError("""dict with keys max_value, adaptive_method,
                                threshold_type, block_size, threshold_const""")

        gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        try:
            adaptive_thresh = cv.adaptiveThreshold(
                src=gray,
                maxValue=adaptive_thresh_params.get('max_value'),
                adaptiveMethod=adaptive_thresh_params.get('adaptive_method'),
                thresholdType=adaptive_thresh_params.get('threshold_type'),
                blockSize=adaptive_thresh_params.get('block_size'),
                C=adaptive_thresh_params.get('threshold_const')
            )
        except cv.error as e:
            print("OpenCV error:", e)
            print("Falling back to standard adaptive threshold parameters")
            adaptive_thresh_params = Frame.adaptive_thresh_params
            adaptive_thresh = cv.adaptiveThreshold(
                src=gray,
                maxValue=adaptive_thresh_params.get('max_value'),
                adaptiveMethod=adaptive_thresh_params.get('adaptive_method'),
                thresholdType=adaptive_thresh_params.get('threshold_type'),
                blockSize=adaptive_thresh_params.get('block_size'),
                C=Frame.adaptive_thresh_params.get('threshold_const')
            )

        return adaptive_thresh

    def enhanced_threshold(
            self,
            kernel_size: int = None,
            adaptive_thresh_params: dict = None) -> np.ndarray:
        """
        Parameters
        ----------
        adaptive_thresh_params : dict, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        morphed : np.ndarray
            Enhanced adaptive threshold.
        """
        if not kernel_size:
            kernel_size = 3
        else:
            if not isinstance(kernel_size, int):
                raise TypeError('kernel_size needs to be an integer')

        adaptive_treshold = self.adaptive_thresh(adaptive_thresh_params)
        morphed = cv.morphologyEx(
            adaptive_treshold,
            cv.MORPH_CLOSE,
            np.ones((kernel_size, kernel_size), np.uint8))
        return morphed

    def maxRect(self, adaptive_thresh_params=None):
        """
        Parameters
        ----------
        adaptive_thresh_params : TYPE, optional
            The default is None.

        Returns
        -------
        max_rect : TYPE
            Returns a the maximal rectangle found in the contours
        """
        if adaptive_thresh_params is None:
            edges = self.adaptive_thresh()
        else:
            edges = self.adaptive_thresh(adaptive_thresh_params)

        contours, _ = cv.findContours(
            edges,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_rect = []
        for contour in contours:
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(
                contour,
                0.04 * perimeter, True)

            if len(approx) == 4:
                x, y, w, h = cv.boundingRect(contour)
                area = float(w)*h
                if area > max_area:
                    if len(max_rect) > 0:
                        max_rect.pop(0)
                    max_rect.append(approx)
        return max_rect
    
    @staticmethod
    def tune_threshold_params(frame: np.ndarray) -> dict:
        """
        Opens an interactive window with trackbars to adjust adaptive threshold
        parameters. Returns the user-defined parameters as a dictionary.
        """
        def nothing(x):
            pass

        # Create a window
        cv.namedWindow('Tuning Adaptive Threshold', cv.WINDOW_NORMAL)

        # Initialize trackbars
        cv.createTrackbar(
            'Block Size',
            'Tuning Adaptive Threshold', 21, 101, nothing)
        cv.createTrackbar(
            'C',
            'Tuning Adaptive Threshold', 5, 50, nothing)
        cv.createTrackbar(
            'Max Value',
            'Tuning Adaptive Threshold', 255, 255, nothing)
        cv.createTrackbar(
            'Threshold Type',
            'Tuning Adaptive Threshold', 0, 1, nothing)
        

        while True:
            # Get current trackbar values
            block_size = cv.getTrackbarPos(
                'Block Size', 'Tuning Adaptive Threshold')
            threshold_const = cv.getTrackbarPos(
                'C', 'Tuning Adaptive Threshold')
            max_value = cv.getTrackbarPos(
                'Max Value', 'Tuning Adaptive Threshold')
            threshold_type = cv.getTrackbarPos(
                'Threshold Type', 'Tuning Adaptive Threshold')

            # Ensure block_size is odd and >= 3
            block_size = max(3, block_size | 1)

            # Define adaptive threshold parameters
            params = {
                'max_value': max_value,
                'adaptive_method': cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                'threshold_type': cv.THRESH_BINARY,
                'block_size': block_size,
                'threshold_const': threshold_const
            }
            
            if threshold_type == 1:
                params['threshold_type'] = cv.THRESH_BINARY_INV

            # Apply adaptive threshold with current parameters
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            processed_frame = cv.adaptiveThreshold(
                src=gray,
                maxValue=params['max_value'],
                adaptiveMethod=params['adaptive_method'],
                thresholdType=params['threshold_type'],
                blockSize=params['block_size'],
                C=params['threshold_const']
            )

            # Display the processed frame
            cv.imshow('Tuning Adaptive Threshold', processed_frame)

            # Break the loop on 'ESC' key
            key = cv.waitKey(10) & 0xFF
            if key == 27:  # ESC key
                break

        cv.destroyAllWindows()
        return params