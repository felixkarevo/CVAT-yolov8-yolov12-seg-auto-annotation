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
    # If a custom model file is available, use it, otherwise use pretrained YOLOv8s-seg
    model_path = "your-custom-yolov8-model.pt"
    if os.path.exists(model_path):
        context.logger.info(f"Loading custom model from {model_path}")
    else:
        model_path = "yolov8s-seg.pt"  # Use pretrained model from Ultralytics
        context.logger.info(f"Custom model not found, using pretrained {model_path}")

    model = YOLO(model_path, task="segment")

    # Read the DL model
    context.user_data.model = model

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run yolo-v8 model")
    try:
        data = event.body
        buf = io.BytesIO(base64.b64decode(data["image"]))
        threshold = float(data.get("threshold", 0.5))
        context.user_data.model.conf = threshold
        image = Image.open(buf)
    except Exception as e:
        context.logger.error(f"Error processing input: {e}")
        return context.Response(body=json.dumps([]), headers={},
                                content_type='application/json', status_code=200)

    try:
        yolo_results = context.user_data.model(image, conf=threshold, imgsz=640, retina_masks=True)[0]
        labels = yolo_results.names
        # Normalize labels to always be a dict {class_id: name}
        if isinstance(labels, (list, tuple)):
            labels = dict(enumerate(labels))
        elif not isinstance(labels, dict):
            context.logger.error(f"Unexpected labels type: {type(labels)}")
            return context.Response(body=json.dumps([]), headers={},
                                    content_type='application/json', status_code=200)
        detections = sv.Detections.from_ultralytics(yolo_results)
        detections = detections[detections.confidence > threshold]
    except Exception as e:
        context.logger.error(f"Error running model inference: {e}")
        return context.Response(body=json.dumps([]), headers={},
                                content_type='application/json', status_code=200)

    results = []
    try:
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
                if len(contours) == 0:
                    context.logger.warning(f"No contours found for detection {i}, skipping")
                    continue

                # Pick the longest contour (outer boundary), not just the first one
                contour = max(contours, key=lambda c: len(c))
                contour = np.flip(contour, axis=1)
                polygons = approximate_polygon(contour, tolerance=1)

                # A valid polygon needs at least 3 vertices
                if len(polygons) < 3:
                    context.logger.warning(
                        f"Polygon too small ({len(polygons)} vertices) for detection {i}, skipping"
                    )
                    continue

                results.append({
                    "confidence": str(confidence),
                    "label": labels.get(class_id, "unknown"),
                    "type": "mask",
                    "points": polygons.ravel().tolist(),
                    "mask": cvat_mask,
                })
    except Exception as e:
        context.logger.error(f"Error building results: {e}")
        # Return empty results on error rather than crashing
        results = []

    try:
        response_body = json.dumps(results)
    except Exception as e:
        context.logger.error(f"Error serializing results: {e}")
        response_body = json.dumps([])

    return context.Response(body=response_body, headers={},
                            content_type='application/json', status_code=200)
