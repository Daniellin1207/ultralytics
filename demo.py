from ultralytics import YOLO
import torch

PYTORCH_CUDA_ALLOC_CONF= True
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

    print(torch.cuda.is_available())

    optimizer = "AdamW"
    epochs = 2010
    train_data = "ultralytics/cfg/datasets/origin.yaml"
    train_data_new = "ultralytics/cfg/datasets/origin_new.yaml"

    ca_yaml = "ultralytics/cfg/models/v8/yolov8-ca.yaml"
    cbam_yaml = "ultralytics/cfg/models/v8/yolov8-cbam.yaml"
    se_yaml = "ultralytics/cfg/models/v8/yolov8-se.yaml"
    yolov8_yaml = "ultralytics/cfg/models/v8/yolov8.yaml"

    cbam_head_yaml =  "ultralytics/cfg/models/v8/yolov8-cbam-head.yaml"
    ca_head_yaml =  "ultralytics/cfg/models/v8/yolov8-ca-head.yaml"
    se_head_yaml =  "ultralytics/cfg/models/v8/yolov8-se-head.yaml"

    p2_yaml = "ultralytics/cfg/models/v8/yolov8s-p2.yaml"
    ghost_yaml = "ultralytics/cfg/models/v8/yolov8s-ghost-p2.yaml"
    yolov5_yaml = "ultralytics/cfg/models/v5/yolov5s.yaml"
    yolov3_yaml = "ultralytics/cfg/models/v3/yolov3s.yaml"

    method_yamls = [ca_yaml,cbam_yaml,se_yaml,yolov8_yaml,cbam_head_yaml,ca_head_yaml,se_head_yaml,p2_yaml,ghost_yaml,yolov5_yaml,yolov3_yaml]
    # method_yamls = [se_head_yaml,se_yaml,yolov8_yaml]
    # train_datas = [train_data,train_data_new]
    optimizers = ["Adam","NAdam","SGD","RAdam","RMSProp","AdamW"] #Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto



    method_yamls = [p2_yaml,ghost_yaml,yolov5_yaml,yolov3_yaml] #cbam_head_yaml,
    train_datas = [train_data]
    # optimizers = ["Adam"]
    for optimizer in optimizers:
        for data in train_datas:
            for yaml in method_yamls:

                try:
                    print("method_yaml:", yaml, "train_data:", data, "running...", optimizer, end="\n\n\n\n", sep="\n")
                    model = YOLO(yaml)

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model.to(device)

                    model.train(data=data, amp=True, epochs=epochs, optimizer=optimizer, workers=6, batch=8,
                                name=optimizer + "_" + data.split('/')[-1].split('.')[0] + "_" +
                                     yaml.split('/')[-1].split('.')[
                                         0] + "_" + str(epochs))
                    print(model.info())
                    print("model info end")
                    print(yaml, data, "running ENDDDDDDD...\n\n")
                except:
                    print("error method_yaml:",yaml,"train_data:",data,"running...",optimizer,end = "\n\n\n\n",sep="\n")

# optimizer = "AdamW"
# print("yolov8-ca is running...train_data_new")
# for i in range(21,13,-1):
#     model = YOLO("ultralytics/cfg/models/v8/yolov8s.yaml")  # build a new model from scratch
#     model.train(data=train_data, epochs=i,optimizer = optimizer,name='yolov8s-train-PRCurve-new_'+str(i))  # train the model
# print("yolov8-ca runs end")




# print("yolov8-ca is running...train_data_new")
# model = YOLO("ultralytics/cfg/models/v8/yolov8-ca.yaml")  # build a new model from scratch
# model.train(data=train_data_new, epochs=epochs,optimizer = optimizer)  # train the model
# print("yolov8-ca runs end")
#
#
# print("yolov8-cbam is running...train_data_new")
# model = YOLO("ultralytics/cfg/models/v8/yolov8-cbam.yaml")  # build a new model from scratch
# model.train(data=train_data_new, epochs=epochs,optimizer = optimizer)  # train the model
# print("yolov8-cbam runs end")
#
#
# print("yolov8-se is running...train_data_new")
# model = YOLO("ultralytics/cfg/models/v8/yolov8-se.yaml")  # build a new model from scratch
# model.train(data=train_data_new, epochs=epochs,optimizer = optimizer)  # train the model
# print("yolov8-se runs end")
#
#
# print("yolov8 is running...train_data_new")
# model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")  # build a new model from scratch
# model.train(data=train_data_new, epochs=epochs,optimizer = optimizer)  # train the model
# print("yolov8 runs end")
#
#
#
# print("yolov8-cbam is running...train_data")
# model = YOLO("ultralytics/cfg/models/v8/yolov8-cbam.yaml")  # build a new model from scratch
# model.train(data=train_data, epochs=epochs,optimizer = optimizer)  # train the model
# print("yolov8-cbam runs end")
#
#
# print("yolov8-se is running...train_data")
# model = YOLO("ultralytics/cfg/models/v8/yolov8-se.yaml")  # build a new model from scratch
# model.train(data=train_data, epochs=epochs,optimizer = optimizer)  # train the model
# print("yolov8-se runs end")
#
#
# print("yolov8 is running...train_data")
# model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")  # build a new model from scratch
# model.train(data=train_data, epochs=epochs,optimizer = optimizer)  # train the model
# print("yolov8 runs end")
#
# print("yolov8-ca is running...train_data")
# model = YOLO("ultralytics/cfg/models/v8/yolov8-ca.yaml")  # build a new model from scratch
# model.train(data=train_data, epochs=epochs,optimizer = optimizer)  # train the model
# print("yolov8-ca runs end")