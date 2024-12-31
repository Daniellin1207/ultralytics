# # 环境参数打印
# import torch
# import cv2
# print("Torch Version:",torch.__version__) #  //注意version前后是两个下划线
# print("OpenCV Version:",cv2.__version__)
# print("Local CUDA Is Available:",torch.cuda.is_available())
# print()
#
# # 计算 # GFlops值
# import torch
# from thop import profile
# from torchvision.models import resnet50
#
# model = resnet50()
# input1 = torch.randn(4, 3, 224, 224)
# flops, params = profile(model, inputs=(input1,))
# print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')


from ultralytics import YOLO
import torch

# 计算 # GFlops值
import torch
from thop import profile
from torchvision.models import resnet50

PYTORCH_CUDA_ALLOC_CONF = True
# Load a model
# model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="ultralytics/cfg/datasets/coco8.yaml", epochs=2000,optimizer = "Adam",amp=False)  # train the model
# model.train(data="ultralytics/cfg/datasets/origin.yaml", epochs=2000)  # train the model

# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format

if __name__ == '__main__':

    train_data = "ultralytics/cfg/datasets/origin.yaml"
    train_data_new = "ultralytics/cfg/datasets/origin_new.yaml"

    ca_yaml = "ultralytics/cfg/models/v8/yolov8-ca.yaml"
    cbam_yaml = "ultralytics/cfg/models/v8/yolov8-cbam.yaml"
    se_yaml = "ultralytics/cfg/models/v8/yolov8-se.yaml"
    yolov8_yaml = "ultralytics/cfg/models/v8/yolov8.yaml"
    csppc_min_yaml = "ultralytics/cfg/models/v8/yolov8-CSPPC.yaml"
    cbam_head_yaml = "ultralytics/cfg/models/v8/yolov8-cbam-head.yaml"
    ca_head_yaml = "ultralytics/cfg/models/v8/yolov8-ca-head.yaml"
    se_head_yaml = "ultralytics/cfg/models/v8/yolov8-se-head.yaml"

    p2_yaml = "ultralytics/cfg/models/v8/yolov8s-p2.yaml"
    ghost_yaml = "ultralytics/cfg/models/v8/yolov8s-ghost-p2.yaml"
    yolov5_yaml = "ultralytics/cfg/models/v5/yolov5s.yaml"
    yolov3_yaml = "ultralytics/cfg/models/v3/yolov3s.yaml"

    method_yamls = [ca_yaml]
    optimizers = ["Adam"]
    train_datas = [train_data]
    epochs = 2000
    for optimizer in optimizers:
        for data in train_datas:
            for yaml in method_yamls:

                try:
                    print("method_yaml:", yaml, "train_data:", data, "running...", optimizer, end="\n\n\n\n", sep="\n")
                    model = YOLO(yaml)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model.to(device)
                    model.train(data=data, amp=True, epochs=epochs, patience=0,optimizer=optimizer, workers=6, batch=8,
                                name=optimizer + "_" + data.split('/')[-1].split('.')[0] + "_" +
                                     yaml.split('/')[-1].split('.')[
                                         0] + "_" + str(epochs))
                    print(model.info())
                    print("model info end")
                    print(yaml, data, "running ENDDDDDDD...\n\n")
                except:
                    print("error method_yaml:", yaml, "train_data:", data, "running...", optimizer, end="\n\n\n\n",
                          sep="\n")


#YOLOv8-CSPPC summary: 216 layers, 2,123,142 parameters, 2,123,126 gradients, 6.0 GFLOPs
