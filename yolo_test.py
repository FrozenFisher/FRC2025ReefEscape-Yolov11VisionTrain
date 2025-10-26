import argparse
from ultralytics import YOLO
import cv2
import torch
import numpy as np

# -----------------------------
# 参数配置
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='YOLO 模型路径（pt 或 SavedModel）')
parser.add_argument('--input_path', type=str, default=None, help='输入路径（视频或图片），不传则使用摄像头')
parser.add_argument('--conf_thresh', type=float, default=0.3, help='置信度阈值')
parser.add_argument('--enable_preprocessing', action='store_true', help='启用预处理')
parser.add_argument('--brightness', type=int, default=0, help='亮度调整值 (-100 到 100)')
parser.add_argument('--contrast', type=float, default=1.0, help='对比度调整值 (0.1 到 3.0)')
parser.add_argument('--gamma', type=float, default=1.0, help='伽马校正值 (0.1 到 3.0)')
args = parser.parse_args()

# -----------------------------
# 图像预处理函数
# -----------------------------
def preprocess_frame(frame, mode='original', brightness=0, contrast=1.0, gamma=1.0):
    """
    对图像进行预处理
    
    Args:
        frame: 输入帧
        mode: 预处理模式
        brightness: 亮度调整值 (-100 到 100)
        contrast: 对比度调整值 (0.1 到 3.0)
        gamma: 伽马校正值 (0.1 到 3.0)
    
    Returns:
        处理后的帧
    """
    if mode == 'original':
        return frame
    
    # 转换为浮点数进行计算
    processed = frame.astype(np.float32)
    
    if mode == 'brightness':
        # 亮度调整
        processed = processed + brightness
        processed = np.clip(processed, 0, 255)
    
    elif mode == 'contrast':
        # 对比度调整
        processed = processed * contrast
        processed = np.clip(processed, 0, 255)
    
    elif mode == 'histogram':
        # 直方图均衡化
        if len(frame.shape) == 3:
            # 彩色图像
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            processed = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            # 灰度图像
            processed = cv2.equalizeHist(frame)
    
    elif mode == 'clahe':
        # 自适应直方图均衡化
        if len(frame.shape) == 3:
            # 转换到LAB色彩空间
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            processed = clahe.apply(frame)
    
    elif mode == 'gamma':
        # 伽马校正
        processed = 255 * np.power(processed / 255.0, gamma)
        processed = np.clip(processed, 0, 255)
    
    return processed.astype(np.uint8)

def is_image_file(file_path):
    """检查文件是否为图片"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    return any(file_path.lower().endswith(ext) for ext in image_extensions)

# -----------------------------
# 设置设备
# -----------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# 加载模型
# -----------------------------
model = YOLO(args.model_path)
model.fuse()
model.to(device)  # 使用 MPS / CPU

# -----------------------------
# 检查输入类型并初始化
# -----------------------------
is_image = False
cap = None
image = None

if args.input_path:
    if is_image_file(args.input_path):
        # 图片输入
        is_image = True
        image = cv2.imread(args.input_path)
        if image is None:
            raise RuntimeError(f"无法读取图片: {args.input_path}")
        print(f"加载图片: {args.input_path}")
    else:
        # 视频输入
        cap = cv2.VideoCapture(args.input_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {args.input_path}")
        print(f"加载视频: {args.input_path}")
else:
    # 摄像头输入
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")
    print("使用摄像头")

# -----------------------------
# 创建窗口
# -----------------------------
cv2.namedWindow("YOLOv11 Detection", cv2.WINDOW_NORMAL)

if args.enable_preprocessing:
    print("预处理功能说明:")
    print("- 按 'q' 键退出")
    print("- 按 's' 键保存当前结果（仅图片模式）")
    print("- 按 '1-4' 键切换预处理模式")
    print("  1: 亮度调整")
    print("  2: 对比度调整") 
    print("  3: 直方图均衡化")
    print("  4: 自适应直方图均衡化")
else:
    print("基本检测模式:")
    print("- 按 'q' 键退出")
    print("- 按 's' 键保存当前结果（仅图片模式）")

# -----------------------------
# 全局变量
# -----------------------------
current_preprocessing_mode = 'brightness'  # 默认预处理模式
preprocessing_modes = {
    '1': ('brightness', '亮度调整'),
    '2': ('contrast', '对比度调整'),
    '3': ('histogram', '直方图均衡化'),
    '4': ('clahe', '自适应直方图均衡化')
}

# -----------------------------
# 处理函数
# -----------------------------
def process_frame(frame):
    """处理单帧图像"""
    if args.enable_preprocessing:
        # 预处理图像
        mode, name = preprocessing_modes.get(current_preprocessing_mode, ('brightness', '亮度调整'))
        processed_frame = preprocess_frame(frame, mode, args.brightness, args.contrast, args.gamma)
        results = model(processed_frame, conf=args.conf_thresh)[0]
        annotated_frame = results.plot()
        
        # 添加预处理信息
        num_detections = len(results.boxes) if results.boxes is not None else 0
        cv2.putText(annotated_frame, f"模式: {name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"检测数量: {num_detections}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        # 无预处理
        results = model(frame, conf=args.conf_thresh)[0]
        annotated_frame = results.plot()
        
        # 添加检测信息
        num_detections = len(results.boxes) if results.boxes is not None else 0
        cv2.putText(annotated_frame, f"检测数量: {num_detections}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # 显示窗口
    cv2.imshow("YOLOv11 Detection", annotated_frame)

# -----------------------------
# 主处理循环
# -----------------------------
if is_image:
    # 图片处理
    print("处理图片...")
    process_frame(image)
    
    # 等待按键
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存结果
            if args.enable_preprocessing:
                mode, name = preprocessing_modes.get(current_preprocessing_mode, ('brightness', '亮度调整'))
                processed_image = preprocess_frame(image, mode, args.brightness, args.contrast, args.gamma)
                results = model(processed_image, conf=args.conf_thresh)[0]
                annotated_frame = results.plot()
                cv2.putText(annotated_frame, f"模式: {name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imwrite(f"result_{mode}.jpg", annotated_frame)
                print(f"保存结果: result_{mode}.jpg")
            else:
                results = model(image, conf=args.conf_thresh)[0]
                annotated_frame = results.plot()
                cv2.imwrite("result.jpg", annotated_frame)
                print("保存结果: result.jpg")
        elif key >= ord('1') and key <= ord('4'):
            # 切换预处理模式
            current_preprocessing_mode = chr(key)
            mode, name = preprocessing_modes.get(current_preprocessing_mode, ('brightness', '亮度调整'))
            print(f"切换到预处理模式: {name}")
            process_frame(image)
else:
    # 视频/摄像头处理
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        process_frame(frame)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif args.enable_preprocessing and key >= ord('1') and key <= ord('4'):
            # 切换预处理模式
            current_preprocessing_mode = chr(key)
            mode, name = preprocessing_modes.get(current_preprocessing_mode, ('brightness', '亮度调整'))
            print(f"切换到预处理模式: {name}")

# 清理资源
if cap is not None:
    cap.release()
cv2.destroyAllWindows()

'''
使用示例:

# 处理图片（带预处理）
python yolo_test.py \
    --model_path /Users/ycc/workspace/FRC/Vision/2025/Yolov11/train_record/frc_reefescapevision_yolov11s_20251026/weights/best.pt \
    --input_path /Users/ycc/workspace/FRC/Vision/2025/Yolov11/train_record/test.png \
    --conf_thresh 0.2 \
    --enable_preprocessing \
    --brightness 20 \
    --contrast 1.2 \
    --gamma 0.8

# 处理视频（带预处理）
python yolo_test.py \
    --model_path /Users/ycc/workspace/FRC/Vision/2025/Yolov11/train_record/frc_reefescapevision_yolov11s_20251026/weights/last.pt \
    --input_path /Users/ycc/workspace/FRC/Vision/2025/Yolov11/train_record/test_video.mp4 \
    --conf_thresh 0.2 \
    --enable_preprocessing \
    --brightness 50 \
    --contrast 1.2 \
    --gamma 0.8

# 使用摄像头（带预处理）
python yolo_test.py \
    --model_path /Users/ycc/workspace/FRC/Vision/2025/Yolov11/train_record/frc_reefescapevision_yolov11s_20251026/weights/best.pt \
    --conf_thresh 0.2 \
    --enable_preprocessing

# 基本使用（无预处理）
python yolo_test.py \
    --model_path /Users/ycc/workspace/FRC/Vision/2025/Yolov11/train_record/frc_reefescapevision_yolov11s_20251026/weights/best.pt \
    --input_path /path/to/image.jpg \
    --conf_thresh 0.2

功能说明:
- 单窗口显示检测结果
- 支持图片和视频输入，自动检测文件类型
- 图片模式: 按 's' 保存结果，按 'q' 退出
- 视频/摄像头模式: 按 'q' 退出

键盘控制（仅预处理模式）:
- '1': 亮度调整
- '2': 对比度调整
- '3': 直方图均衡化
- '4': 自适应直方图均衡化
- 's': 保存结果（仅图片模式）
- 'q': 退出

参数说明:
- --brightness: 亮度调整值 (-100 到 100)
- --contrast: 对比度调整值 (0.1 到 3.0)  
- --gamma: 伽马校正值 (0.1 到 3.0)

'''