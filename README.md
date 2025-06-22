# Human Segmentation Using Classical Machine Learning

This project implements a lightweight, interpretable solution for human segmentation in images using a Decision Tree classifier, entirely without deep learning. It prioritizes computational efficiency and transparency, making it suitable for constrained environments.

Key Features:
- Pixel-wise Human Segmentation using handcrafted features from multiple color spaces (RGB, LAB, HSV, YCrCb, Grayscale) and spatial coordinates.
- Lightweight Model: Trained Decision Tree is compiled into pure Python logic for deployment without any ML library overhead.
- Morphological Post-processing: Enhances mask quality using classical image processing techniques (opening/closing).
- Performance: Achieves an average F1 score of 0.7699 on 128×128 resolution test images using a Decision Tree of depth 7.

Technologies Used
- Python, NumPy, OpenCV
- Scikit-learn (for training only, not used in final inference code)
