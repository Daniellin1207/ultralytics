datasets = ""


from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8-cbam.yaml")  # build a new model from scratch
# model = YOLO("yolov8-se.yaml")  # build a new model from scratch
# model = YOLO("yolov8-ca.yaml")  # build a new model from scratch
model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")  # build a new model from scratch



# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="coco8.yaml", epochs=3)  # train the model
model.train(data="ultralytics/cfg/datasets/origin.yaml", epochs=2000)  # train the model
# model.train(data="ultralytics/cfg/datasets/origin.yaml", epochs=3)  # train the model


# ultralytics/cfg/datasets/origin.yaml
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format