import cv2
import numpy as np

def load_yolo_model():
    net = cv2.dnn.readNet("C:/Users/aarya/OneDrive/Desktop/Assert.ai/yolov5/yolov5s.pt", "C:/Users/aarya/OneDrive/Desktop/Assert.ai/yolov5/.venv/Lib/site-packages/ultralytics/yolo/cfg/default.yaml")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, classes, output_layers

def detect_objects(image_path, net, classes, output_layers):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return class_ids, confidences, boxes

def draw_labels(image, class_ids, confidences, boxes, classes):
    for i in range(len(class_ids)):
        class_id = class_ids[i]
        confidence = confidences[i]
        box = boxes[i]
        label = f"{classes[class_id]}: {confidence:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
        cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

if __name__ == "__main__":
    image_path = "C:/Users/aarya/Downloads/office.jpeg"
    
    net, classes, output_layers = load_yolo_model()
    class_ids, confidences, boxes = detect_objects(image_path, net, classes, output_layers)

    image = cv2.imread(image_path)
    labeled_image = draw_labels(image.copy(), class_ids, confidences, boxes, classes)

    cv2.imshow("YOLO Object Detection", labeled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
