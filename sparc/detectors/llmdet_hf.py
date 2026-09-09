from sparc.detectors._batching import create_batches
import numpy as np
import supervision as sv
import torch

import torchvision
from tqdm import tqdm
from transformers import (
    AutoModelForZeroShotObjectDetection,
    GroundingDinoProcessor,
)
from transformers.image_transforms import center_to_corners_format
from transformers.models.grounding_dino.processing_grounding_dino import (
    get_phrases_from_posmap,
)


device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomGroundingDinoProcessor(GroundingDinoProcessor):
    def post_process_grounded_object_detection(
        self,
        outputs,
        input_ids=None,
        threshold: float = 0.25,
        text_threshold: float = 0.25,
        target_sizes=None,
        token_span_lens=None,
        reduce_threshold=False,
        classes=None,
    ):
        """Converts the raw output of [`GroundingDinoForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format and get the associated text label.

        Args:
            outputs ([`GroundingDinoObjectDetectionOutput`]):
                Raw outputs of the model.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The token ids of the input text.
            threshold (`float`, *optional*, defaults to 0.25):
                Score threshold to keep object detection predictions.
            text_threshold (`float`, *optional*, defaults to 0.25):
                Score threshold to keep text detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        box_threshold = threshold
        logits, boxes = outputs.logits, outputs.pred_boxes
        input_ids = input_ids if input_ids is not None else outputs.input_ids

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        probabilities = torch.sigmoid(logits)  # (batch_size, num_queries, 256)
        scores = torch.max(probabilities, dim=-1)[0]  # (batch_size, num_queries)

        # Convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(boxes)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, list):
                image_heights = torch.Tensor([i[0] for i in target_sizes])
                image_widths = torch.Tensor([i[1] for i in target_sizes])
            else:
                image_heights, image_widths = target_sizes.unbind(1)

            scale_factors = torch.stack(
                [image_widths, image_heights, image_widths, image_heights], dim=1
            ).to(boxes.device)
            boxes = boxes * scale_factors[:, None, :]

        results = []

        if token_span_lens is not None:
            for image_index, (
                query_scores,
                query_boxes,
                token_probabilities,
            ) in enumerate(zip(scores, boxes, probabilities)):
                last_start_idx = 1

                keep = query_scores > box_threshold
                query_scores = query_scores[keep]
                query_boxes = query_boxes[keep]
                token_probabilities = token_probabilities[keep]

                batch_scores, batch_boxes, batch_labels = get_results_from_token_scores(
                    query_scores,
                    query_boxes,
                    token_probabilities,
                    token_span_lens,
                    text_threshold,
                    last_start_idx,
                )

                res_dict = {
                    "scores": batch_scores,
                    "labels": batch_labels,
                    "boxes": batch_boxes,
                }

                results.append(res_dict)

                if len(results[-1]["boxes"]) == 0 and reduce_threshold:
                    batch_scores, batch_boxes, batch_labels = (
                        get_results_from_token_scores(
                            query_scores,
                            query_boxes,
                            token_probabilities,
                            token_span_lens,
                            box_threshold / 2,
                            last_start_idx,
                        )
                    )
                    res_dict = {
                        "scores": batch_scores,
                        "labels": batch_labels,
                        "boxes": batch_boxes,
                    }
                    results[-1] = res_dict

                if len(results[-1]["boxes"]) > 0:
                    results[-1]["scores"] = torch.stack(results[-1]["scores"]).cpu()
                    results[-1]["boxes"] = torch.stack(results[-1]["boxes"]).cpu()
                    results[-1]["labels"] = results[-1]["labels"]

                    if len(classes) == 3:
                        # find robot in labels and reduce score by half
                        robot_indices = [
                            i
                            for i, label in enumerate(results[-1]["labels"])
                            if "robot" in classes[label]
                        ]
                        results[-1]["scores"][robot_indices] *= 0.5

                    if len(results[-1]["boxes"]) > 0:
                        keep = torchvision.ops.nms(
                            results[-1]["boxes"], results[-1]["scores"], 0.7
                        )
                        results[-1]["scores"] = results[-1]["scores"][keep]
                        results[-1]["boxes"] = results[-1]["boxes"][keep]
                        results[-1]["labels"] = [results[-1]["labels"][i] for i in keep]
                else:
                    results[-1] = {
                        "scores": torch.tensor([]),
                        "labels": torch.tensor([]),
                        "boxes": torch.tensor([]),
                    }

            for result in results:
                result["labels"] = [classes[i] for i in result["labels"]]
                result["text_labels"] = result["labels"]

            return results

        else:
            for image_index, (
                query_scores,
                query_boxes,
                token_probabilities,
            ) in enumerate(zip(scores, boxes, probabilities)):
                score = query_scores[query_scores > box_threshold]
                box = query_boxes[query_scores > box_threshold]
                prob = token_probabilities[query_scores > box_threshold]
                label_ids = get_phrases_from_posmap(
                    prob > text_threshold, input_ids[image_index]
                )
                label = self.batch_decode(label_ids)
                results.append({"scores": score, "labels": label, "boxes": box})

        return results


def get_results_from_token_scores(
    s, b, p, token_span_lens, box_threshold, last_start_idx
):
    selected_scores = []
    selected_boxes = []
    selected_class_ids = []

    for class_id in range(len(token_span_lens)):
        token_start = last_start_idx
        token_end = last_start_idx + token_span_lens[class_id]

        token_probs = p[:, token_start:token_end]

        mean_token_scores = torch.mean(token_probs, dim=-1)

        class_box_indices = mean_token_scores > box_threshold
        class_boxes = b[class_box_indices]
        class_scores = mean_token_scores[class_box_indices]

        for box, score in zip(class_boxes, class_scores):
            selected_scores.append(score)
            selected_boxes.append(box)
            selected_class_ids.append(class_id)

        last_start_idx = token_end + 1

    return selected_scores, selected_boxes, selected_class_ids


class LLMDet:
    def __init__(
        self,
        model_id="iSEE-Laboratory/llmdet_base",
        box_threshold=0.35,
        text_threshold=0.25,
        use_slicing=False,
    ):

        self.use_slicing = use_slicing

        self.processor = CustomGroundingDinoProcessor.from_pretrained(model_id)
        self.detection_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(device)

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.separator_id = 1012

    def detect_objects(
        self,
        images,
        classes,
        threshold=0.24,
        bsz=4,
        reduce_threshold=False,
        use_slicing=False,
        text_threshold=0.24,
    ):

        self.detection_model = self.detection_model.to(device)

        if use_slicing:

            def inference_callback(image):

                boxes = []
                scores = []
                labels = []

                queries = ""
                for query in classes:
                    queries += f"{query}. "

                prompt = ". ".join(classes)
                prompt = [queries]
                model_inputs = self.processor(
                    images=image, text=prompt, return_tensors="pt"
                ).to(device)

                with torch.inference_mode():
                    outputs = self.detection_model(**model_inputs)

                separator_indices = torch.where(
                    model_inputs.input_ids[0] == self.separator_id
                )[0].cpu()
                class_token_spans = list(
                    model_inputs.input_ids[0].cpu().tensor_split(separator_indices)
                )
                token_span_lens = [len(span) - 1 for span in class_token_spans]
                token_span_lens = token_span_lens[:-1]

                target_sizes = [image.shape]

                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    model_inputs.input_ids,
                    box_threshold=threshold,
                    text_threshold=threshold,
                    target_sizes=target_sizes,
                    token_span_lens=token_span_lens,
                    reduce_threshold=reduce_threshold,
                )[0]

                boxes = results["boxes"].cpu().numpy()
                scores = results["scores"].cpu().numpy()
                labels = results["labels"]

                boxes = np.array(boxes).reshape(-1, 4)
                scores = np.array(scores).reshape(-1)
                labels = np.array(labels).reshape(-1)

                detections = sv.Detections(
                    xyxy=boxes, confidence=scores, class_id=labels
                )

                return detections

            slicer = sv.InferenceSlicer(
                callback=inference_callback,
                slice_wh=(64, 64),
                overlap_ratio_wh=(0.2, 0.2),
                iou_threshold=0.5,
                overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
                thread_workers=1,
            )
            detections = []
            for img in tqdm(images):
                image_detections = slicer(img)
                detections.append(image_detections)
            return detections

        if isinstance(images, np.ndarray):
            images = torch.tensor(images).permute(0, 3, 1, 2).float()
        if isinstance(images, torch.Tensor):
            if images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)

        images_split = create_batches(bsz, images)


        boxes = []
        scores = []
        labels = []

        queries = ""
        for query in classes:
            queries += f"{query}. "

        for i in tqdm(range(len(images_split)), disable=True):
            image = images_split[i]

            prompt = ". ".join(classes)
            prompt = [classes] * len(image)
            model_inputs = self.processor(
                images=image, text=prompt, return_tensors="pt"
            ).to(device)

            with torch.inference_mode():
                outputs = self.detection_model(**model_inputs)
            separator_indices = torch.where(
                model_inputs.input_ids[0] == self.separator_id
            )[0].cpu()
            class_token_spans = list(
                model_inputs.input_ids[0].cpu().tensor_split(separator_indices)
            )
            token_span_lens = [len(span) - 1 for span in class_token_spans]
            token_span_lens = token_span_lens[:-1]

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                threshold=threshold,
                target_sizes=[image.shape[2:]] * len(image),
                token_span_lens=token_span_lens,
                text_threshold=text_threshold,
                classes=classes,
            )

            for result in results:
                valid_indices = []
                label_indices = []
                for label in result["labels"]:
                    if label in classes:
                        label_indices.append(classes.index(label))
                        valid_indices.append(True)
                    else:
                        valid_indices.append(False)
                detected_classes = [
                    result["labels"][idx] for idx, i in enumerate(valid_indices) if i
                ]
                non_robot_count = sum(
                    1 for cls in detected_classes if "robot" not in cls
                )
                if non_robot_count == 0 and np.sum(valid_indices) <= len(valid_indices):
                    valid_indices = []
                    label_indices = []

                    for label in result["labels"]:
                        matched = False
                        for class_name in classes:
                            if label in class_name:
                                label_indices.append(classes.index(class_name))
                                valid_indices.append(True)
                                matched = True
                                break
                        if not matched:
                            valid_indices.append(False)

                boxes.append(
                    result["boxes"].cpu().numpy().reshape(-1, 4)[valid_indices]
                )
                scores.append(result["scores"].cpu().numpy()[valid_indices])
                labels.append(np.array(label_indices))

        detections = [
            sv.Detections(xyxy=box, confidence=score, class_id=np.array(label))
            for box, score, label in zip(boxes, scores, labels)
        ]

        return detections

    def resize(self, image, size):
        return self.detection_processor.image_processor.resize(image, size)
