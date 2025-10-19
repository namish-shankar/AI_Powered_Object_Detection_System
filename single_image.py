from ultralytics import YOLO

model = YOLO('runs/detect/train7/weights/best.pt')  

image_path = '/mnt/34B471F7B471BBC4/CSO_project/datasets/test_dataset/test2017/000000000069.jpg'  

results = model(image_path) 

results[0].show()

results[0].save(filename='test_image_predicted.jpg')

for box in results[0].boxes:
    class_id = int(box.cls[0].cpu().numpy())
    confidence = float(box.conf[0].cpu().numpy())
    xyxy = box.xyxy[0].cpu().numpy()  
    print(f"Class: {model.names[class_id]}, Confidence: {confidence:.2f}, Box: {xyxy}")
