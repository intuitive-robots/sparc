import matplotlib
matplotlib.use("Agg", force=True)

from matplotlib import patches
import matplotlib.pyplot as plt


import numpy as np
import copy
import supervision as sv


def plot_boxes_np(image, boxes, labels=None, scores=None, return_image=True):

    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Create a Rectangle patch for each box and add it to the axes
    for i, box in enumerate(boxes):
        if box.sum() == 0:
            continue

        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Add the score and label text next to the box if they are provided
        if scores is not None:
            if labels is None:
                labels = [""] * len(boxes)
            plt.text(x1, y1, f'{labels[i]}: {round((scores[i]), 2):.2f}', bbox=dict(facecolor='white', alpha=0.5))

    if not return_image:
        plt.show()
    else:
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.tight_layout()
        plt.axis('off')
        ax.axis('off')
        ax.set_axis_off()
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close()

        return image


def get_top_k_boxes(detections, k = 1):


    filtered_detections = []
    for detection in detections:
        if detection is None:

            filtered_detections.append(sv.Detection(xyxy=np.array([].reshape(0, 4)), confidence=np.array([]), class_id=np.array([])))
            continue
        filtered_detection = copy.deepcopy(detection)
        if detection.xyxy.shape[0] > 0:
            top_k_indices = np.argsort(detection.confidence)[-k:]

            filtered_detection.xyxy = detection.xyxy[top_k_indices]
            if detection.confidence is not None:
                filtered_detection.confidence = detection.confidence[top_k_indices]
            if detection.class_id is not None:
                filtered_detection.class_id = detection.class_id[top_k_indices]


        filtered_detections.append(filtered_detection)

    return filtered_detections
