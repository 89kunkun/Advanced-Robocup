import cv2
import numpy as np
from ultralytics import YOLO


# ===============================
# 1. 加载 YOLOv8 Seg 模型
# ===============================
model = YOLO("yolov8s-seg.pt")

# ===============================
# 2. 打开摄像头
# ===============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    # ===============================
    # 3. 推理（Seg）
    # ===============================
    results = model(frame, conf=0.4, iou=0.5)
    r = results[0]

    # ===============================
    # 4. 可视化（bbox + mask）
    # ===============================
    annotated_frame = r.plot()   # YOLO自带：透明 mask 覆盖

    # ===============================
    # 5. 处理每个实例的 mask
    # ===============================
    if r.masks is not None:
        masks = r.masks.data.cpu().numpy() # (N, H, W)
        boxes = r.boxes

        for i in range(masks.shape[0]):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            name = model.names.get(cls_id, str(cls_id))

            # 取第 i 个实例 mask(bool)
            mask = masks[i] > 0.5  # (H, W)

            # --- 5.1 画 mask 轮廓（更清晰） ---
            mask_u8 = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(
                mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(annotated_frame, contours, -1, (0, 255, 0), 2)

            # --- 5.2 计算 mask 中心（像素级） ---
            ys, xs = np.where(mask)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())

                cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(
                    annotated_frame,
                    f"{name} {conf:.2f}",
                    (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

    # ===============================
    # 6. 显示
    # ===============================
    cv2.imshow("YOLOv8 Seg Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
