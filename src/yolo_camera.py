import torch
import cv2

# Model​
model = torch.hub.load("ultralytics/yolov5", "yolov5m")

# Video capture
cap = cv2.VideoCapture(0)

# TODO: Loop for camera frames
def main_loop():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yolo_image_rect(frame)
        cv2.imshow("YOLOv5 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# # Read frame (BGR to RGB)
# ret, frame = cap.read()
# # TODO: break the loop on error
# if not ret:
#     break

def yolo_image_rect(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    
    # TODO: Boudning box 그리기
    for i, obj in enumerate(results.xyxy[0]):
        # TODO: 인식결과를 표시하기 위한 좌표를 얻음
        x1, y1, x2, y2 = map(int, obj[:4])
        confidence = obj[4]
        label = f"{model.names[int(obj[5])]}: {confidence:.2f}"

        # TODO: 인식된 정확도(confidence)와 클래스를 label로 구성
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # TODO: OpenCV를 이용해서 해당 좌표에 사각형과 text를 출력
        obj_info = list(map(int, obj))
        print(f"Object {i}: {model.names[obj_info[5]]}")
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



# TODO: 화면 표시

# TODO: 종료를 위한 key 처리

# cap.release()
# cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()