import cv2
import numpy as np

class YoloDetector:
    def __init__(self, model_path='yolov5s.onnx', class_id=0, conf_threshold=0.45, nms_threshold=0.45):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.class_id = class_id # 0 = person
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def detect_and_isolate(self, img):
        """
        Runs YOLOv5 on the image to find the person.
        Returns:
            isolated_img: An image containing ONLY the person pixels (rest is black).
                          This guarantees MediaPipe only sees the isolated subject.
            visual_img: The original image with purely the bounding box drawn on it.
            detected: Boolean whether a person was found.
        """
        original_img = img.copy()
        height, width = img.shape[:2]
        
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()
        
        # preds shape is (1, 25200, 85)
        # Parse detections
        outputs = preds[0]
        boxes = []
        confidences = []
        
        # Object confidence is output[:, 4]
        # Class 0 confidence is output[:, 5]
        
        # Multiply obj_conf and class_conf
        scores = outputs[:, 4] * outputs[:, 5]
        mask = scores > self.conf_threshold
        
        valid_outputs = outputs[mask]
        valid_scores = scores[mask]
        
        for i, detection in enumerate(valid_outputs):
            cx, cy, w, h = detection[0:4]
            # scale back to original image
            x_scale = width / 640
            y_scale = height / 640
            
            x_center = cx * x_scale
            y_center = cy * y_scale
            box_width = w * x_scale
            box_height = h * y_scale
            
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            boxes.append([x1, y1, int(box_width), int(box_height)])
            confidences.append(float(valid_scores[i]))
            
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        isolated_img = np.zeros_like(img) # Create empty/black image
        if len(indices) > 0:
            # Flatten indices depending on opencv version output
            indices = indices.flatten()
            best_idx = indices[0]
            x, y, bw, bh = boxes[best_idx]
            
            # Add 20% spatial padding so we don't chop off hands/feet
            pad_w = int(bw * 0.20)
            pad_h = int(bh * 0.20)
            
            # Constrain padded bounds to prevent index errors
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(width, x + bw + pad_w)
            y2 = min(height, y + bh + pad_h)
            
            # Mask the person out (isolate with padding)
            isolated_img[y1:y2, x1:x2] = original_img[y1:y2, x1:x2]
            
            # Draw overlay to visualize YOLO boundaries (the raw box, not padded)
            # using the original non-padded x, y, bw, bh for the HUD display
            rx1, ry1 = max(0, x), max(0, y)
            rx2, ry2 = min(width, x + bw), min(height, y + bh)
            cv2.rectangle(original_img, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
            cv2.putText(original_img, "YOLOv5 Isolated Person", (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            return isolated_img, original_img, True
            
        return img, original_img, False
