# Use the official Python image from Docker Hub
FROM python:3.11.9-slim-bullseye

ADD object_detection.py .

RUN pip install --no-cache-dir torch pillow

CMD [ "python", "./object_detection.py" ]
