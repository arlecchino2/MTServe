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
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Callable, Tuple

import torch


class NegativesSampler(torch.nn.Module):
    """"""

    def __init__(self, norm_func: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self._normalizer = norm_func

    def normalize_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalizer(x)

    @abc.abstractmethod
    def forward(
        self,
        positive_ids: torch.Tensor,
        num_to_sample: int,
        sampled_candiate_ids: torch.Tensor,
        sampled_candiate_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple of (sampled_ids, sampled_negative_embeddings).
        """


class InBatchNegativesSampler(NegativesSampler):
    def __init__(
        self,
        norm_func: Callable[[torch.Tensor], torch.Tensor],
        dedup_embeddings: bool,
    ) -> None:
        super().__init__(norm_func)

        self._dedup_embeddings: bool = dedup_embeddings

    def gen_sampled_candidates(
        self,
        valid_ids: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
           valid_ids: (T,)
           embeddings: (T, D) x float
        """
        assert valid_ids.numel() == embeddings.size(0)
        if self._dedup_embeddings:
            unique_ids, unique_ids_inverse_indices = torch.unique(
                input=valid_ids, sorted=False, return_inverse=True
            )
            device = unique_ids.device
            indices = torch.empty(
                (unique_ids.numel(),),
                dtype=torch.int64,
                device=device,
            )
            indices[unique_ids_inverse_indices] = torch.arange(
                valid_ids.numel(), dtype=torch.int64, device=device
            )
            unique_embeddings = embeddings[indices, :]
            sampled_candiate_embeddings = self.normalize_embeddings(unique_embeddings)
            sampled_candiate_ids = unique_ids
        else:
            sampled_candiate_embeddings = self.normalize_embeddings(embeddings)
            sampled_candiate_ids = valid_ids
        return sampled_candiate_ids, sampled_candiate_embeddings

    def forward(
        self,
        positive_ids: torch.Tensor,
        num_to_sample: int,
        sampled_candidate_ids: torch.Tensor,
        sampled_candidate_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple of (sampled_ids, sampled_negative_embeddings,).
        """
        X = sampled_candidate_ids.size(0)
        if X == 0:
            empty_indices = torch.empty(
                size=positive_ids.size() + (num_to_sample,),
                dtype=positive_ids.dtype,
                device=positive_ids.device,
            )
            empty_embeddings = sampled_candidate_embeddings[empty_indices]
            return empty_indices, empty_embeddings

        sampled_indices = torch.randint(
            low=0,
            high=X,
            size=positive_ids.size() + (num_to_sample,),
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        )

        return (
            sampled_candidate_ids[sampled_indices],
            sampled_candidate_embeddings[sampled_indices],
        )
