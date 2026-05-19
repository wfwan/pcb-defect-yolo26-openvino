import cv2
import numpy as np
import openvino as ov
import sys

CONF_THRESHOLD = 0.25
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


def draw_detections(frame, detections, orig_w, orig_h, model_w, model_h):
    # Scale boxes back to original frame size
    x_scale = orig_w / model_w
    y_scale = orig_h / model_h

    for det in detections:
        x1, y1, x2, y2, confidence, class_id = det
        if confidence > CONF_THRESHOLD:
            class_id = int(class_id)
            label = CLASS_NAMES[class_id]
            color = COLORS[class_id]
            text = f"{label} {confidence:.2f}"

            x1 = int(x1 * x_scale)
            y1 = int(y1 * y_scale)
            x2 = int(x2 * x_scale)
            y2 = int(y2 * y_scale)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            cv2.putText(
                frame,
                text,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    return frame


def main():
    core = ov.Core()
    model = core.read_model("best.xml")
    compiled_model = core.compile_model(model, "CPU")
    n, c, model_h, model_w = compiled_model.input(0).shape

    # Change this to 0 for webcam, or a path like "video.mp4"
    video_path = "pcb_defects.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        return 1

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output video writer
    out = cv2.VideoWriter(
        "out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (orig_w, orig_h)
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        tensor = preprocess(frame, model_w, model_h)

        results = compiled_model.infer_new_request({0: tensor})
        predictions = next(iter(results.values()))
        detections = predictions[0]

        frame = draw_detections(frame, detections, orig_w, orig_h, model_w, model_h)

        # Show FPS on frame
        cv2.putText(
            frame,
            f"Frame: {frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        out.write(frame)

        # Optional: show live preview (remove if running headless)
        cv2.imshow("PCB Defect Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done. Processed {frame_count} frames → out.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
