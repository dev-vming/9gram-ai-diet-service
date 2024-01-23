from ultralytics import YOLO

def process_img(img):
    model = YOLO('ai/best.pt')
    result = model(img)[0].boxes

    return [result.cls.tolist(), result.xywhn.tolist()]
