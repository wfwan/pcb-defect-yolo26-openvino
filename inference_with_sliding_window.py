import cv2
import numpy as np
import openvino as ov
import sys

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
WINDOW_SIZE = 600
OVERLAP = 0.2

CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]
COLORS = [
    (0, 255, 0),  # missing_hole - green
    (255, 0, 0),  # mouse_bite - blue
    (0, 165, 255),  # open_circuit - orange
    (0, 0, 255),  # short - red
    (255, 255, 0),  # spur - cyan
    (255, 0, 255),  # spurious_copper - magenta
]


def preprocess(frame, width, height):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (width, height))
    tensor = resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    return tensor


def sliding_window_inference(image, compiled_model, model_w, model_h):
    orig_h, orig_w = image.shape[:2]
    step = int(WINDOW_SIZE * (1 - OVERLAP))
    all_detections = []

    # Handle images smaller than window size
    if orig_h <= WINDOW_SIZE and orig_w <= WINDOW_SIZE:
        tensor = preprocess(image, model_w, model_h)
        results = compiled_model.infer_new_request({0: tensor})
        detections = next(iter(results.values()))[0]
        x_scale = orig_w / model_w
        y_scale = orig_h / model_h
        for det in detections:
            x1, y1, x2, y2, confidence, class_id = det
            if confidence > CONF_THRESHOLD:
                all_detections.append(
                    [
                        x1 * x_scale,
                        y1 * y_scale,
                        x2 * x_scale,
                        y2 * y_scale,
                        confidence,
                        class_id,
                    ]
                )
        return np.array(all_detections)

    y_coords = list(range(0, orig_h - WINDOW_SIZE, step)) + [orig_h - WINDOW_SIZE]
    x_coords = list(range(0, orig_w - WINDOW_SIZE, step)) + [orig_w - WINDOW_SIZE]

    for y in y_coords:
        for x in x_coords:
            crop = image[y : y + WINDOW_SIZE, x : x + WINDOW_SIZE]
            tensor = preprocess(crop, model_w, model_h)

            results = compiled_model.infer_new_request({0: tensor})
            detections = next(iter(results.values()))[0]

            # Scale from model size to window size
            x_scale = WINDOW_SIZE / model_w
            y_scale = WINDOW_SIZE / model_h

            for det in detections:
                x1, y1, x2, y2, confidence, class_id = det
                if confidence > CONF_THRESHOLD:
                    all_detections.append(
                        [
                            x1 * x_scale + x,
                            y1 * y_scale + y,
                            x2 * x_scale + x,
                            y2 * y_scale + y,
                            confidence,
                            class_id,
                        ]
                    )

    return np.array(all_detections)


def apply_nms(detections):
    if len(detections) == 0:
        return []
    boxes = detections[:, :4].tolist()
    scores = detections[:, 4].tolist()
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
    if len(indices) == 0:
        return []
    return detections[indices.flatten()]


def draw_detections(image, detections):
    for det in detections:
        x1, y1, x2, y2, confidence, class_id = det
        class_id = int(class_id)
        label = CLASS_NAMES[class_id]
        color = COLORS[class_id]
        text = f"{label} {confidence:.2f}"

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(
            image, (int(x1), int(y1) - th - 8), (int(x1) + tw, int(y1)), color, -1
        )
        cv2.putText(
            image,
            text,
            (int(x1), int(y1) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


def main():

    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <path_to_image>')
        return 1

    image_path = sys.argv[1] 

    core = ov.Core()
    model = core.read_model("best.xml")
    compiled_model = core.compile_model(model, "CPU")
    n, c, model_h, model_w = compiled_model.input(0).shape

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: cannot open image {image_path}")
        return 1

    orig_h, orig_w = image.shape[:2]
    print(f"Image size: {orig_w}x{orig_h}")

    detections = sliding_window_inference(image, compiled_model, model_w, model_h)
    print(f"Raw detections: {len(detections)}")

    if len(detections) > 0:
        detections = apply_nms(detections)
        print(f"After NMS: {len(detections)}")
        image = draw_detections(image, detections)

        for det in detections:
            x1, y1, x2, y2, confidence, class_id = det
            print(
                f"  {CLASS_NAMES[int(class_id)]}: {confidence:.2f} @ ({int(x1)},{int(y1)}) ({int(x2)},{int(y2)})"
            )
    else:
        print("No detections found.")

    cv2.imwrite("out.jpg", image)
    print("Saved → out.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
