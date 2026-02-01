import cv2
from ultralytics import YOLO

# 使用官方小模型（速度快）
model = YOLO("yolov8s.pt")

# 打开本机摄像头（/dev/video0）
cap = cv2.VideoCapture(0)

# 设置分辨率（可选）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    # YOLOv8 推理
    results = model(frame)

    # 将识别结果绘制到图像上
    annotated_frame = results[0].plot()

    # 显示画面
    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
