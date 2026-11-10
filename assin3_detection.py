# ===============================
# Assignment 3-2: Object Detection Threshold Comparison
# ===============================

import jetson.inference
import jetson.utils

# 1. 自带模型（无需 fruit onnx）
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.3)

# 2. 两张测试图路径
img_path_1 = "image1.jpeg"
img_path_2 = "image2.jpeg"

# 3. 检测函数：重新读图 + 检测 + 打印 + 保存
def run_detection(img_path, output_name, threshold_name):
    image = jetson.utils.loadImage(img_path)   # 防框叠加
    print("=" * 60)
    print(f"Threshold: {threshold_name}")
    print("=" * 60)

    detections = net.Detect(image)

    print(f"Number of detections: {len(detections)}")
    for i, detection in enumerate(detections):
        print(f"\nDetection {i + 1}:")
        print(f"  Class ID     : {detection.ClassID}")
        print(f"  Confidence   : {detection.Confidence:.3f}")
        print(f"  Left         : {detection.Left:.2f}")
        print(f"  Top          : {detection.Top:.2f}")
        print(f"  Right        : {detection.Right:.2f}")
        print(f"  Bottom       : {detection.Bottom:.2f}")
        print(f"  Width        : {detection.Width:.2f}")
        print(f"  Height       : {detection.Height:.2f}")
        print(f"  Area         : {detection.Area:.2f}")
        print(f"  Center       : {detection.Center}")

    jetson.utils.saveImageRGBA(output_name, image)  # 保存带框图
    print(f"Saved -> {output_name}")
    return detections

# 4. 低阈值 0.3
net.SetThreshold(0.3)
det_img1_low = run_detection(img_path_1, "img1_low_0.3.png", "0.3")
det_img2_low = run_detection(img_path_2, "img2_low_0.3.png", "0.3")

# 5. 高阈值 0.6
net.SetThreshold(0.6)
det_img1_high = run_detection(img_path_1, "img1_high_0.6.png", "0.6")
det_img2_high = run_detection(img_path_2, "img2_high_0.6.png", "0.6")

# 6. 对比总结
print("\n" + "=" * 60)
print("Comparison Summary")
print("=" * 60)
print(f"Image 1 low threshold 0.3 detections : {len(det_img1_low)}")
print(f"Image 1 high threshold 0.6 detections: {len(det_img1_high)}")
print(f"Image 2 low threshold 0.3 detections : {len(det_img2_low)}")
print(f"Image 2 high threshold 0.6 detections: {len(det_img2_high)}")
print("\nConclusion:")
print("Lower threshold keeps more boxes, but may include false positives.")
print("Higher threshold keeps fewer boxes, but may miss true objects.")
