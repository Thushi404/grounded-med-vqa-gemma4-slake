"""SLAKE data preparation: link detection.json boxes to questions and build targets.

SLAKE stores boxes per image (imgs/<folder>/detection.json), not per question.
Each box is [x, y, width, height] in pixels. This module decides which of an
image's boxes, if any, ground each question.
"""

import json
import os
import re
from pathlib import PurePosixPath
from zipfile import ZipFile

# detection.json labels that describe a finding rather than normal anatomy.
DISEASE_LABELS = {
    "atelectasis",
    "brain edema",
    "brain enhancing tumor",
    "brain non-enhancing tumor",
    "cardiomegaly",
    "effusion",
    "infiltrate",
    "liver cancer",
    "lung cancer",
    "mass",
    "nodule",
    "pneumonia",
    "pneumothorax",
}

# Answers which say the object is absent or not shown.
NEGATIVE_ANSWERS = {"no", "none", "not seen", "0"}

# Words which make a yes/no question ask whether an object is present.
PRESENCE_WORDS = ("contain", "is there", "are there", "exist", "observed", "seen", "visible", "present")

# A finding is present when the answer is not "no" ("Is the lung abnormal?").
ABNORMAL_PATTERN = re.compile(r"\b(abnormal\w*|disease\w*|lesions?)\b")

# A finding is present when the answer is "no" ("Is the lung healthy?").
NORMAL_PATTERN = re.compile(r"\b(healthy|normal)\b")

# Question types where an object named in the question should be grounded.
# Modality, Plane and KG questions do not ask about a place in the image.
QUESTION_GROUNDED_TYPES = {"Organ", "Position", "Size", "Shape", "Color", "Abnormality", "Quantity"}

SIDE_PATTERN = re.compile(r"^(left|right) ")


def _contains_phrase(text, phrase):
    """True when phrase appears in text as whole words, singular or plural."""
    return re.search(r"\b" + re.escape(phrase) + r"(s|es)?\b", text) is not None


def find_mentioned_labels(text, labels):
    """Return the labels that text refers to.

    A full name ("left kidney") always matches. A label without its side
    ("kidney") matches every side of that organ, but only when the text does
    not already name one side. This keeps "Where is the left lung?" from
    also matching the right lung.
    """
    text = text.lower()
    lowered = [label.lower() for label in labels]
    full_matches = {label for label in lowered if _contains_phrase(text, label)}
    bases_with_full_match = {SIDE_PATTERN.sub("", label) for label in full_matches}

    mentioned = set(full_matches)
    for label in lowered:
        base = SIDE_PATTERN.sub("", label)
        if base != label and base not in bases_with_full_match and _contains_phrase(text, base):
            mentioned.add(label)
        if "tumor" in label and (_contains_phrase(text, "brain tumor") or _contains_phrase(text, "tumor")):
            mentioned.add(label)

    return [label for label in labels if label.lower() in mentioned]


def match_regions(row, detections):
    """Return the boxes that ground one SLAKE question.

    Rules, checked in this order:
      1. A presence or "where/which/what" question with a negative answer
         gets no box.
      2. A question which says a finding is present ("Where is the
         abnormality?", "Is the lung healthy?" -> "No") gets the disease
         boxes named in the answer, then in the question, else every disease
         box in the image.
      3. Otherwise, objects named in the answer get their boxes
         ("What is the largest organ?" -> "Liver").
      4. Otherwise, objects named in the question get their boxes, for the
         question types in QUESTION_GROUNDED_TYPES.

    Returns a list of {"label", "bbox_xywh_pixels"} dicts in detection.json
    order. An empty list means the question has no box.
    """
    question = row["question"].lower()
    answer = row["answer"].strip().lower()
    objects = [(label, box) for item in detections for label, box in item.items()]
    labels = list(dict.fromkeys(label for label, _ in objects))

    if answer in NEGATIVE_ANSWERS:
        is_presence_question = any(word in question for word in PRESENCE_WORDS)
        if is_presence_question or question.startswith(("where", "which", "what")):
            return []

    if ABNORMAL_PATTERN.search(question):
        finding_present = answer != "no"
    elif NORMAL_PATTERN.search(question):
        finding_present = answer == "no"
    else:
        finding_present = False

    disease_labels = [label for label in labels if label.lower() in DISEASE_LABELS]
    chosen = []
    if finding_present and disease_labels:
        chosen = (
            find_mentioned_labels(answer, disease_labels)
            or find_mentioned_labels(question, disease_labels)
            or disease_labels
        )
    if not chosen:
        chosen = find_mentioned_labels(answer, labels)
    if not chosen and row.get("content_type") in QUESTION_GROUNDED_TYPES:
        chosen = find_mentioned_labels(question, labels)

    regions = []
    seen = set()
    for label, box in objects:
        key = (label, tuple(box))
        if label in chosen and key not in seen:
            seen.add(key)
            regions.append({"label": label, "bbox_xywh_pixels": list(box)})
    return regions


def build_records(rows, imgs_zip_path, image_dir="/content/slake_images"):
    """Extract each row's image and attach its matched regions.

    Returns (records, missing_images). The archive is indexed once, so this
    stays fast for every split.
    """
    records = []
    missing_images = []

    with ZipFile(imgs_zip_path) as archive:
        archive_files = [name.replace("\\", "/") for name in archive.namelist()]
        images = {}
        detection_files = {}
        for name in archive_files:
            path = PurePosixPath(name)
            if path.name == "detection.json":
                detection_files[path.parent.name] = name
            elif len(path.parts) >= 2:
                images[f"{path.parent.name}/{path.name}"] = name

        detections_cache = {}

        for row in rows:
            image_name = row["img_name"]
            folder = PurePosixPath(image_name).parent.as_posix()

            image_zip_path = images.get(image_name)
            if image_zip_path is None:
                missing_images.append(image_name)
                continue

            image_path = os.path.join(image_dir, image_name)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            if not os.path.exists(image_path):
                with open(image_path, "wb") as image_file:
                    image_file.write(archive.read(image_zip_path))

            if folder not in detections_cache:
                detection_zip_path = detection_files.get(folder)
                detections_cache[folder] = (
                    json.loads(archive.read(detection_zip_path)) if detection_zip_path else []
                )

            records.append({
                "img_id": row["img_id"],
                "img_name": image_name,
                "image_path": image_path,
                "qid": row["qid"],
                "question": row["question"],
                "answer": row["answer"],
                "content_type": row.get("content_type"),
                "regions": match_regions(row, detections_cache[folder]),
            })

    return records, missing_images


def to_target_text(record, image_width, image_height):
    """Build the JSON string the model learns to output.

    Boxes become [x1, y1, x2, y2] normalized to 0..1 and clipped to the image.
    A question without a box gets "regions": [].
    """
    regions = []
    for region in record["regions"]:
        x, y, width, height = region["bbox_xywh_pixels"]
        x1 = min(max(x / image_width, 0.0), 1.0)
        y1 = min(max(y / image_height, 0.0), 1.0)
        x2 = min(max((x + width) / image_width, 0.0), 1.0)
        y2 = min(max((y + height) / image_height, 0.0), 1.0)
        if x2 > x1 and y2 > y1:
            regions.append({
                "label": region["label"],
                "bbox_xyxy_normalized": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
            })

    return json.dumps({"answer": record["answer"], "regions": regions}, separators=(",", ":"))
