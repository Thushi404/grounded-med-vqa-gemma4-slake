import json

from src.data import find_mentioned_labels, match_regions, to_target_text

ABDOMEN = [
    {"Liver": [54.0, 106.0, 30.0, 31.0]},
    {"Left Kidney": [160.0, 120.0, 20.0, 25.0]},
    {"Right Kidney": [70.0, 120.0, 20.0, 25.0]},
    {"Liver Cancer": [60.0, 110.0, 10.0, 10.0]},
]
CHEST = [
    {"Left Lung": [130.0, 40.0, 90.0, 150.0]},
    {"Right Lung": [30.0, 40.0, 90.0, 150.0]},
    {"Pneumonia": [40.0, 120.0, 30.0, 30.0]},
]
BRAIN = [
    {"Brain Edema": [124.0, 98.0, 45.0, 68.0]},
    {"Brain Enhancing Tumor": [128.0, 108.0, 33.0, 39.0]},
    {"Brain Enhancing Tumor": [140.0, 150.0, 8.0, 12.0]},
]


def labels_for(question, answer, detections, content_type="Organ"):
    row = {"question": question, "answer": answer, "content_type": content_type}
    return [region["label"] for region in match_regions(row, detections)]


def test_presence_question_follows_answer():
    assert labels_for("Does the picture contain liver?", "Yes", ABDOMEN) == ["Liver"]
    assert labels_for("Does the picture contain liver?", "No", ABDOMEN) == []


def test_location_answer_is_not_limited_to_left_or_right():
    assert labels_for("Where is the liver?", "Upper Left", ABDOMEN, "Position") == ["Liver"]
    assert labels_for("Where is the liver?", "Not seen", ABDOMEN, "Position") == []


def test_organ_without_side_matches_both_sides():
    assert labels_for("How many kidneys in this image?", "2", ABDOMEN, "Quantity") == [
        "Left Kidney",
        "Right Kidney",
    ]


def test_named_side_matches_only_that_side():
    assert find_mentioned_labels("where is the left lung?", ["Left Lung", "Right Lung"]) == ["Left Lung"]


def test_object_named_in_answer_is_grounded():
    assert labels_for("What is the largest organ in the picture?", "Liver", ABDOMEN, "Size") == ["Liver"]


def test_disease_question_uses_disease_named_in_answer():
    assert labels_for("What diseases are included in the picture?", "Pneumonia", CHEST, "Abnormality") == [
        "Pneumonia"
    ]


def test_abnormality_location_uses_disease_box_not_organ_box():
    labels = labels_for("Where is/are the abnormality located?", "Liver", ABDOMEN, "Position")
    assert labels == ["Liver Cancer"]


def test_healthy_question_depends_on_answer():
    assert labels_for("Is the lung healthy?", "No", CHEST, "Abnormality") == ["Pneumonia"]
    assert labels_for("Is the liver healthy?", "Yes", [{"Liver": [1, 1, 5, 5]}], "Abnormality") == ["Liver"]


def test_every_instance_of_a_label_is_kept():
    labels = labels_for("What diseases are included in the picture?", "Brain Enhancing Tumor", BRAIN, "Abnormality")
    assert labels == ["Brain Enhancing Tumor", "Brain Enhancing Tumor"]


def test_question_without_image_region_has_no_box():
    assert labels_for("What modality is used to take this image?", "MRI", ABDOMEN, "Modality") == []


def test_target_text_normalizes_and_clips_boxes():
    record = {
        "answer": "Yes",
        "regions": [
            {"label": "Liver", "bbox_xywh_pixels": [54.0, 106.0, 30.0, 31.0]},
            {"label": "Edge", "bbox_xywh_pixels": [250.0, 250.0, 20.0, 20.0]},
        ],
    }
    target = json.loads(to_target_text(record, 256, 256))
    assert target["regions"][0] == {"label": "Liver", "bbox_xyxy_normalized": [0.2109, 0.4141, 0.3281, 0.5352]}
    assert target["regions"][1]["bbox_xyxy_normalized"] == [0.9766, 0.9766, 1.0, 1.0]


def test_target_text_without_box_has_empty_regions():
    assert to_target_text({"answer": "MRI", "regions": []}, 256, 256) == '{"answer":"MRI","regions":[]}'
