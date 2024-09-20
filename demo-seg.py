from ultralytics import YOLO
import cv2
# 加载模型
from DrawLinesTask import OperateImg
# model = YOLO('yolov8n-seg.yaml').load('seg-best.pt')  # 从YAML构建并转移权重
model = YOLO('seg-best.pt')  # 从YAML构建并转移权重

if __name__ == '__main__':
    # 训练模型
    # results = model.train(data='ultralytics/cfg/datasets/origin-seg.yaml', epochs=2000, imgsz=512)
    # metrics = model.val()
    # 预测结果
    local_save_img = "local1.png"
    predict_img = "test1.jpg"
    results = model.predict(predict_img,save = True)
    print(results)
    # 处理结果列表
    img = None
    for result in results:
        boxes = result.boxes  # Boxes 对象，用于边界框输出
        masks = result.masks  # Masks 对象，用于分割掩码输出
        keypoints = result.keypoints  # Keypoints 对象，用于姿态输出
        probs = result.probs  # Probs 对象，用于分类输出

        if result.masks is not None and len(result.masks) > 0:
            masks_data = result.masks.data
            for index, mask in enumerate(masks_data):
                mask = mask.cpu().numpy() * 255
                cv2.imwrite(local_save_img, mask)
                if img is not None:
                    img+=mask
                else:
                    img = mask

    tmp = OperateImg(local_save_img,None)
    tmp.OperateContours()
    cv2.imshow("mask",tmp.imageCopy)
    cv2.waitKey()



    # print("boxes输出示意：\n", boxes)
    # print("masks输出示意：\n", masks)
    # print("keypoints输出示意：\n", keypoints)
    # print("probs输出示意：\n", probs)