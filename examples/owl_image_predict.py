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


import argparse
import PIL.Image
import time
import torch
import cv2
from nanoowl.owl_predictor import (
    OwlPredictor
)
from nanoowl.owl_drawing import (
    markup_image
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="../assets/owl_glove_small.jpg")
    parser.add_argument("--query", type=str, default="../assets/owl_target_1.jpg")
    parser.add_argument("--threshold", type=str, default="0.25")
    parser.add_argument("--output", type=str, default="/root/out/owl_image_guided_out.jpg")
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--num_profiling_runs", type=int, default=30)
    args = parser.parse_args()

    threshold = float(args.threshold)
    print(threshold)
    

    predictor = OwlPredictor(
        args.model,
    )

    image = PIL.Image.open(args.image)
    query_image = [cv2.imread(args.query)]

    output = predictor.detect_from_img(
        image=image, 
        query_imgs=query_image, 
        threshold=threshold,
        pad_square=False
    )

    if args.profile:
        torch.cuda.current_stream().synchronize()
        t0 = time.perf_counter_ns()
        for i in range(args.num_profiling_runs):
            output = predictor.detect_from_img(
                image=image, 
                query_imgs=query_image, 
                threshold=threshold,
                pad_square=False
            )
        torch.cuda.current_stream().synchronize()
        t1 = time.perf_counter_ns()
        dt = (t1 - t0) / 1e9
        print(f"PROFILING FPS: {args.num_profiling_runs/dt}")

    image = markup_image(image, output, draw_text=True)

    image.save(args.output)