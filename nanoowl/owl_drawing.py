# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import PIL.Image
import PIL.ImageDraw
import cv2
from .owl_predictor import OwlDecodeOutput
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional


def get_colors(count: int):
    cmap = plt.get_cmap("rainbow", count)
    colors = []
    for i in range(count):
        color = cmap(i)
        color = [int(255 * value) for value in color]
        colors.append(tuple(color))
    return colors


def markup_image(image, output: OwlDecodeOutput, text: Optional[List[str]] = None, draw_text=True, redact=False):
    is_pil = isinstance(image,PIL.Image.Image)
    if is_pil:
        image = np.array(image)

    if redact:
        h,w = image.shape[:2]
        image = cv2.resize(image, (32,32), interpolation=cv2.INTER_LINEAR)
        image = cv2.resize(image, (w,h), interpolation=cv2.INTER_NEAREST)

    num_detections = len(output.labels)
    if num_detections == 0:
        return image


    if not text:
        if draw_text:
            text = [f"object-{idx}" for idx in output.labels]
        else:
            draw_text = False

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    colors = get_colors(np.max(output.labels) + 1)

    for i in range(num_detections):
        box = output.boxes[i]
        label_index = int(output.labels[i])
        iou = float(output.scores[i])
        box = [int(x) for x in box]
        pt0 = (box[0], box[1])
        pt1 = (box[2], box[3])

        cv2.rectangle(image,pt0,pt1,colors[label_index],4)
        cv2.rectangle(image, (box[0]-2, box[1]-40), (box[0]+300, box[1]), colors[label_index], -1)
        cv2.putText(image, f"{iou:.2f} |", (box[0]+10, box[1]-10), font, font_scale, (255,255,255),2)

        if draw_text:
            label_text = text[label_index]
            cv2.putText(image,label_text,(box[0] + 115, box[1] -10),font,font_scale,(255,255,255),2)

    if is_pil:
        image = PIL.Image.fromarray(image)

    return image