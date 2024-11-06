import os

from ultralytics import YOLO
import torch
import time

def InitTorchEnvs(model_file):
    PYTORCH_CUDA_ALLOC_CONF= True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_file)
    model.to(device)
    return model

def GetPredictImgs(folder_path):
    import os
    jpg_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jpg'):
                jpg_paths.append(os.path.join(root, file))
    return jpg_paths

def PredictImgs(model:YOLO,imgs:list):
    i = 0
    for img in imgs:
        results = model.predict(img)
        # print(results)
        print(i)
        i+=1
    return

if __name__ == '__main__':

    data_path = "E:\DemoDatas\datasets\origin\images\\train"
    paths = GetPredictImgs(data_path)

    Adam_origin_yolov5s_2010 = "E:\DemoDatas\\runs\detect\Adam_origin_yolov5s_2010\weights"
    Adam_origin_yolov3s_2010 = "E:\DemoDatas\\runs\detect\Adam_origin_yolov3s_2010\weights"
    Adam_origin_yolov8s_ghost_p2_2010 = "E:\DemoDatas\\runs\detect\Adam_origin_yolov8s-ghost-p2_2010\weights"
    Adam_origin_yolov8s_p2_2010 = "E:\DemoDatas\\runs\detect\Adam_origin_yolov8s-p2_2010\weights"
    cbam_head_origin_2000 = "E:\DemoDatas\\runs\detect\cbam-head-origin-2000\weights"

    model_files = [Adam_origin_yolov5s_2010,Adam_origin_yolov3s_2010,Adam_origin_yolov8s_ghost_p2_2010,Adam_origin_yolov8s_p2_2010,cbam_head_origin_2000]
    i = 0
    for model_file in model_files:
        model_file = model_file+"\\best.pt"
        model = InitTorchEnvs(model_file)
        begin_time = time.time()
        PredictImgs(model,paths)
        PredictImgs(model,paths)
        PredictImgs(model,paths)
        PredictImgs(model,paths)
        final_time = time.time()
        use_time = final_time-begin_time
        print(model_file," use time:",80/use_time,i)
        i += 1


# E:\DemoDatas\runs\detect\cbam-head-origin-2000\weights\best.pt  use time: 113.51323227084774
# E:\DemoDatas\runs\detect\Adam_origin_yolov8s-p2_2010\weights\best.pt  use time: 116.61412599612844
# E:\DemoDatas\runs\detect\Adam_origin_yolov8s-ghost-p2_2010\weights\best.pt  use time: 95.99503809827334
# E:\DemoDatas\runs\detect\Adam_origin_yolov3s_2010\weights\best.pt  use time: 52.78505790251549
# E:\DemoDatas\runs\detect\Adam_origin_yolov5s_2010\weights\best.pt  use time: 60.09167112295156
