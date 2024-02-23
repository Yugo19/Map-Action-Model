from ultralytics import YOLO
from PIL import Image
import cv2

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("/home/mapaction/mapaction_env/Map-Action-Model/code/map_action_yolo_detection/runs/detect/train5/weights/best.pt")  # load a pretrained model (recommended for training)


model.train(data="data.yaml", epochs=20)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("/home/mapaction/Documents/Exp-data/YOLO/images/dumpgarbage103_jpeg.rf.fe62527242b552a1d2ad0f12f83dacd4.jpg")  # predict on an image
path = model.export(format="onnx")

#img1 = Image.open("/home/mapaction/Documents/Exp-data/YOLO/images/dumpgarbage103_jpeg.rf.fe62527242b552a1d2ad0f12f83dacd4.jpg")

#results = model.predict(source=img1, save=True)

for r in results:
    im_array = r.plot()
    im = Image.fromarray(im_array[...,::-1])
    im.show()
    im.save('results.jpg')