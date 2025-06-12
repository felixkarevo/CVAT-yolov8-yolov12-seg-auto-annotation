import json
import base64
from PIL import Image
import io
import os

import numpy as np
from ultralytics import YOLO
import supervision as sv
from skimage.measure import approximate_polygon, find_contours


def to_cvat_mask(box: list, mask):
    xtl, ytl, xbr, ybr = box
    flattened = mask[ytl:ybr + 1, xtl:xbr + 1].flat[:].tolist()
    flattened.extend([xtl, ytl, xbr, ybr])
    return flattened


def init_context(context):
    context.logger.info("Init context...  0%")

    # Check for custom model or use pretrained model
    # If a custom model file is available, use it, otherwise use a pretrained YOLOv12s-seg
    model_path = "your-custom-yolov12-model.pt"
    if os.path.exists(model_path):
        context.logger.info(f"Loading custom model from {model_path}")
    else:
        model_path = "yolov12s-seg.pt"  # Use pretrained model from Ultralytics
        context.logger.info(f"Custom model not found, using pretrained {model_path}")

    model = YOLO(model_path, task="segment")

    # Read the DL model
    context.user_data.model = model

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run yolo-v12 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    context.user_data.model.conf = threshold
    image = Image.open(buf)

    yolo_results = context.user_data.model(image, conf=threshold)[0]
    labels = yolo_results.names
    detections = sv.Detections.from_ultralytics(yolo_results)
    detections = detections[detections.confidence > threshold]

    results = []
    if len(detections) > 0:
        for i in range(len(detections)):
            xyxy = detections.xyxy[i]
            mask = detections.mask[i]
            confidence = detections.confidence[i]
            class_id = detections.class_id[i]

            mask = mask.astype(np.uint8)

            xtl = int(xyxy[0])
            ytl = int(xyxy[1])
            xbr = int(xyxy[2])
            ybr = int(xyxy[3])

            label = int(class_id)
            cvat_mask = to_cvat_mask((xtl, ytl, xbr, ybr), mask)

            contours = find_contours(mask, 0.5)
            contour = contours[0]
            contour = np.flip(contour, axis=1)
            polygons = approximate_polygon(contour, tolerance=2.5)

            results.append({
                "confidence": str(confidence),
                "label": labels.get(class_id, "unknown"),
                "type": "mask",
                "points": polygons.ravel().tolist(),
                "mask": cvat_mask,
            })

    return context.Response(body=json.dumps(results), headers={},
                            content_type='application/json', status_code=200)