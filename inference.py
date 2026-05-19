import cv2
import numpy as np
import openvino as ov
import sys

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

def main():

    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <path_to_image>')
        return 1

    image_path = sys.argv[1]    
    # Initiate OpenVINO Runtime Core
    core = ov.Core()

    # Read a model
    model = core.read_model("best.xml")

    compiled_model = core.compile_model(model, "CPU")
    n, c, height, width = compiled_model.input(0).shape

    # Set up input
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (width, height))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    input_tensor = image_rgb.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)

    results = compiled_model.infer_new_request({0: input_tensor})
    predictions = next(iter(results.values()))

    output = predictions[0]
    for detection in output:
        x1, y1, x2, y2, confidence, class_id = detection

        if confidence > 0.5:
            class_id = int(class_id)
            label = CLASS_NAMES[class_id]
            color = COLORS[class_id]
            text = f"{label} {confidence:.2f}"
            print(f"Found: {label}, confidence={confidence:.2f}")
            print(f"coords=({int(x1), {int(y1)}}), ({int(x2)}, {int(y2)})")
            cv2.rectangle(
                image_resized, (int(x1), int(y1)), (int(x2), int(y2)), color, 2
            )

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(
                image_resized,
                (int(x1), int(y1) - th - 8),
                (int(x1) + tw, int(y1)),
                color,
                -1,
            )
            cv2.putText(
                image_resized,
                text,
                (int(x1), int(y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite("out.jpg", image_resized)

    return 0


if __name__ == "__main__":
    sys.exit(main())
