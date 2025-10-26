import cv2
import numpy as np
import argparse

# -----------------------------
# 参数配置
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='tf', choices=['tf', 'openvino'],
                    help='模型类型: tf (TensorFlow SavedModel) 或 openvino (OpenVINO FP16)')
parser.add_argument('--model_path', type=str, required=True, help='模型路径')
parser.add_argument('--video_path', type=str, default=None, help='视频文件路径，不传则使用摄像头')
parser.add_argument('--imgsz', type=int, default=512, help='模型输入尺寸')
parser.add_argument('--conf_thresh', type=float, default=0.3, help='置信度阈值')
args = parser.parse_args()

# -----------------------------
# Letterbox 缩放
# -----------------------------
def letterbox(img, new_shape=(512,512)):
    """Resize image with unchanged aspect ratio using padding"""
    h0, w0 = img.shape[:2]
    r = min(new_shape[0]/h0, new_shape[1]/w0)
    new_unpad = int(w0*r), int(h0*r)
    dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
    dw /= 2
    dh /= 2
    img_resized = cv2.resize(img, new_unpad)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    return img_padded, r, dw, dh

# -----------------------------
# 初始化模型
# -----------------------------
if args.model_type == 'tf':
    import tensorflow as tf
    print("Loading TensorFlow SavedModel...")
    model = tf.saved_model.load(args.model_path)
    infer = model.signatures["serving_default"]
    is_tf = True
elif args.model_type == 'openvino':
    from openvino.runtime import Core
    print("Loading OpenVINO FP16 model...")
    ie = Core()
    model_ir = ie.read_model(model=args.model_path + "/model.xml")
    compiled_model = ie.compile_model(model_ir, device_name="CPU")  # 可改为 GPU
    infer_request = compiled_model.create_infer_request()
    is_tf = False
else:
    raise ValueError("Unsupported model type")

# -----------------------------
# 打开视频/摄像头
# -----------------------------
cap = cv2.VideoCapture(args.video_path) if args.video_path else cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("无法打开视频或摄像头")

# -----------------------------
# 推理循环
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h0, w0 = frame.shape[:2]
    img_resized, r, dw, dh = letterbox(frame, new_shape=(args.imgsz, args.imgsz))
    input_tensor = np.expand_dims(img_resized, axis=0).astype(np.float32)

    # -----------------------------
    # 推理
    # -----------------------------
    if is_tf:
        outputs = infer(tf.convert_to_tensor(input_tensor))
        key = list(outputs.keys())[0]
        detections = outputs[key].numpy()[0]  # [N,6]
    else:
        outputs = infer_request.infer({"input": input_tensor})
        key = list(outputs.keys())[0]
        detections = outputs[key][0]  # [N,6]

    # -----------------------------
    # 处理输出
    # -----------------------------
    if detections.shape[0] == 0:
        boxes, scores, classes = np.zeros((0,4)), np.zeros((0,)), np.zeros((0,),dtype=int)
    else:
        boxes = detections[:, :4]
        scores = detections[:, 4]
        classes = detections[:, 5].astype(int)

    # -----------------------------
    # 恢复到原图坐标
    # -----------------------------
    boxes[:, [0,2]] -= dw
    boxes[:, [1,3]] -= dh
    boxes /= r

    # 绘制检测框
    for box, score, cls in zip(boxes, scores, classes):
        if score < args.conf_thresh:
            continue
        x1, y1, x2, y2 = box.astype(int)
        label = f"{cls}:{score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # 显示窗口
    cv2.imshow("YOLOv11 Detection", frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



'''
python test.py \
    --model_type tf \
    --model_path /Users/ycc/workspace/FRC/Vision/2025/Yolov11/train_record/frc_reefescapevision_yolov11s_20251026/weights/best_saved_model \
    --video_path /Users/ycc/workspace/FRC/Vision/2025/Yolov11/train_record/test_video.mp4 \
    --imgsz 512 \
    --conf_thresh 0.3


python test.py \
    --model_type openvino \
    --model_path /Users/you/YOLO_Projects/frc_reefescapevision_yolov11s/weights/openvino_fp16 \
    --video_path /Users/you/test_video.mp4 \
    --imgsz 512 \
    --conf_thresh 0.3
'''