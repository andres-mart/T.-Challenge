import torch
from PIL import Image
import json

# Model setup
model = torch.hub.load("ultralytics/yolov5", "yolov5s")
model.classes = [0, 2]  # 0: person, 2: car


# Image upload
def upload_image():
    # Get image path from user
    while True:
        image_path = input("Enter the path to your image (or 'q' to quit): ")
        if image_path.lower() == "q":
            return None

        try:
            # Open the image using PIL
            img = Image.open(image_path)
            return img
        except FileNotFoundError:
            print("Invalid image path. Please try again.")


# Object detection
def detect_objects(image):
    # Perform object detection
    results = model(image)

    df = results.pandas().xyxy[0]

    person_count = df[df["name"] == "person"].shape[0]
    car_count = df[df["name"] == "car"].shape[0]

    output = {"number_of_people": person_count, "number_of_cars": car_count}

    print(json.dumps(output, indent=4))


if __name__ == "__main__":
    user_image = upload_image()
    if user_image:
        detect_objects(user_image)
    else:
        print("Exiting...")
