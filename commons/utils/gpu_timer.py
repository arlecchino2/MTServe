# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import statistics
from collections import defaultdict

import torch


class GPUTimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record()

    def stop(self):
        self.end_event.record()

    def elapsed_time(self):
        """
        return in ms
        """
        self.dist_sync()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)

    def dist_sync(self):
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])


class IGPUTimer(GPUTimer):
    def __init__(self, max_iters=1):
        self._max_iters = max_iters

        self.start_events = [
            torch.cuda.Event(enable_timing=True) for i in range(max_iters)
        ]
        self.end_events = [
            torch.cuda.Event(enable_timing=True) for i in range(max_iters)
        ]

        self._recorded_start_events = defaultdict()
        self._recorded_end_events = defaultdict()

    def start(self, ith=0):
        self.start_events[ith].record()
        self._recorded_start_events[ith] = True

    def stop(self, ith=0):
        self.end_events[ith].record()
        self._recorded_end_events[ith] = True

    def elapsed_time(self, reduction="mean"):
        self.dist_sync()
        torch.cuda.synchronize()
        times = []
        for idx in self._recorded_start_events.keys():
            assert idx in self._recorded_end_events, "end_event is not recorded "
            times.append(self.start_events[idx].elapsed_time(self.end_events[idx]))

        self._recorded_start_events.clear()
        self._recorded_end_events.clear()
        if reduction == "mean":
            ret_time = sum(times) / len(times)
        elif reduction == "max":
            ret_time = max(times)
        elif reduction == "median":
            ret_time = statistics.median(times)
        else:
            raise ValueError(f"reduction {reduction} is not supported")

        return ret_time

    def reset(self):
        self._recorded_start_events.clear()
        self._recorded_end_events.clear()

    def dist_sync(self):
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
