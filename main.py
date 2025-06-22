import os
import cv2
import numpy as np
from sklearn.metrics import f1_score
from abc import *

RESIZE_SHAPE = (64, 64)

def resize_image(img, size=RESIZE_SHAPE):
    try:
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"Error resizing image: {e}")
        return np.zeros(size)  # Return an empty image of the same size

def extract_features(img):
    try:
        h, w = img.shape[:2]
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = np.concatenate([
            img.reshape(-1, 3),
            img_lab.reshape(-1, 3),
            img_ycrcb.reshape(-1, 3),
            img_hsv.reshape(-1, 3),
            img_gray.reshape(-1, 1)
        ], axis=1)

        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        coords = np.stack([yy, xx], axis=-1).reshape(-1, 2)
        features = np.concatenate([features, coords], axis=1)

        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros((1, 7))  # Return empty features

def decision_tree_classifier(x):
    try:
        if x[4] <= -0.14:
            if x[3] <= -0.89:
                if x[13] <= -0.32:
                    if x[13] <= -0.81:
                        if x[14] <= -0.54:
                            if x[13] <= -0.87:
                                return 0
                            else:
                                if x[7] <= -0.35:
                                    return 0
                                else:
                                    return 0
                        else:
                            if x[14] <= 0.38:
                                if x[11] <= -1.42:
                                    return 0
                                else:
                                    return 0
                            else:
                                if x[14] <= 0.97:
                                    return 0
                                else:
                                    return 0
                    else:
                        if x[14] <= 1.08:
                            if x[5] <= -0.30:
                                if x[14] <= -1.08:
                                    return 0
                                else:
                                    return 1
                            else:
                                if x[14] <= 0.11:
                                    return 0
                                else:
                                    return 0
                        else:
                            if x[5] <= -0.05:
                                if x[14] <= 1.19:
                                    return 0
                                else:
                                    return 0
                            else:
                                if x[7] <= -0.35:
                                    return 0
                                else:
                                    return 1
                else:
                    if x[5] <= -0.59:
                        if x[3] <= -0.99:
                            return 1
                        else:
                            return 0
                    else:
                        return 1
            else:
                if x[7] <= 1.16:
                    if x[14] <= -0.50:
                        return 0
                    else:
                        return 1
                else:
                    return 0
        else:
            if x[14] <= -0.66:
                if x[13] <= -0.37:
                    return 0
                else:
                    return 1
            else:
                if x[7] <= -0.35:
                    return 0
                else:
                    return 1
    except Exception as e:
        print(f"Error in decision tree classification: {e}")
        return 0  # Return a default value (0)

class HumanSegmentationMapGenerator:
    def predict_segmentation_map(self, img):
        try:
            original_shape = img.shape[:2]
            img_resized = resize_image(img)
            features = extract_features(img_resized)
            
            preds = np.array([decision_tree_classifier(f) for f in features], dtype=np.uint8)
            mask_resized = preds.reshape(img_resized.shape[:2]) * 255

            kernel = np.ones((3, 3), np.uint8)
            mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_OPEN, kernel)
            mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)

            mask = cv2.resize(mask_resized, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
            return mask
        except Exception as e:
            print(f"Error in predicting segmentation map: {e}")
            return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)  # Return an empty mask if any error occurs
