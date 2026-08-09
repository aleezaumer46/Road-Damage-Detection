from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")

# Run prediction
results = model.predict(
    source="test_images/China_Drone_000008.jpg",
    save=True,
    conf=0.25
)

# Display detected objects
for result in results:
    print("\nDetected Objects:")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        print(f"Class: {class_name}")
        print(f"Confidence: {confidence * 100:.2f}%")