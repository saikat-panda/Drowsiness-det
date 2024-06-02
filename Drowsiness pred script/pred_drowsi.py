import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

labels = ['Awake', 'Drowsy']
font_scale=0.7
color=(255, 255, 255)
thickness=2

class ObjectDetection:
    def __init__(self, capture_index):
        self.model = self.load_model()
        self.capture_index = capture_index

    def load_model(self):
        model = YOLO(r"C:\Users\sdadi\Desktop\Work\ML\Drowsiness_detec\yolov8s_drowsy.pt") #PAth of the model.pt file 
        model.fuse()  
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def classify(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []
        color = (0, 255, 0)  # Green for bounding boxes

        try:
            result =results[0]
            boxes = result.boxes.cpu().numpy()
            # print(boxes)
            # xyxy=boxes.xyxy
            # print(xyxy)
            argmax= np.argmax(boxes.conf[0])
            labelmax= boxes.cls[argmax]
            text= labels[int(labelmax)]
            print(text)
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            # # Calculate the bottom center coordinates for the text
            text_offset_x = int((frame.shape[1] - text_width) / 2)
            text_offset_y = frame.shape[0] - int(text_height * 1.5)  # Adjust for some padding below
            
            # # Add the text overlay to the frame
            cv2.putText(frame, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # for xyxy in xyxys:
            # frame= cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
            # label= f"{self.model.names[class_ids[-1]]} {confidences[-1]:.2f}"
            # frame= cv2.putText(frame, label, (xyxys[-1][0], xyxys[-1][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # print(boxes)
            # xmin, ymin, xmax, ymax, conf, cls = result.tolist()
            # xyxys.append([int(xmin), int(ymin), int(xmax), int(ymax)])
            # confidences.append(conf)
            # class_ids.append(int(cls))

            # # Draw bounding boxes and labels
            # if self.model.names:  # Check if class names are available
            #     label = f"{self.model.names[class_ids[-1]]} {confidences[-1]:.2f}"
        except:
            pass
        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened(), "Error opening webcam!"

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame from webcam!")
                break
            
            frame1 = cv2.resize(frame, (640, 640))
            results = self.predict(frame1)
            # frame= cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = self.classify(results, frame)
            # frame1= cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = ObjectDetection(0)  
    detector()
