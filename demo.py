from ultralytics import YOLO

# Load a model
# model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="ultralytics/cfg/datasets/coco8.yaml", epochs=2000,patience=300)  # train the model
# model.train(data="ultralytics/cfg/datasets/origin.yaml", epochs=2000)  # train the model

# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format