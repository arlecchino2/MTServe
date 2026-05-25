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

import torch
import torch.nn.functional as F
from modules.negatives_sampler import NegativesSampler
from modules.similarity.dot_product import DotProductSimilarity


class AutoregressiveLoss(torch.nn.Module):
    """"""

    @abc.abstractmethod
    def forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            output_embeddings: [B, N, D] x float, embeddings for the current
                input sequence.
            supervision_ids: [B, N] x int64, (positive) supervision ids.
            supervision_embeddings: [B, N, D] x float.
        Returns:
            (1), loss for the current engaged sequence.
        """


class SampledSoftmaxLoss(AutoregressiveLoss):
    def __init__(
        self,
        num_to_sample: int,
        softmax_temperature: float,
        negatives_sampler: NegativesSampler,
        interaction_module: DotProductSimilarity,
        activation_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self._num_to_sample: int = num_to_sample
        self._softmax_temperature: float = softmax_temperature
        self._activation_checkpoint: bool = activation_checkpoint
        self._negatives_sampler = negatives_sampler
        self._interaction_module = interaction_module

    # To calculate the loss, follow below steps
    # 1. do negative sampling as negative_embeddings
    # 2. calculate the similarity between user_embeddings and supervision_embeddings as positive logits [BxN]
    # 3. calculate the similarity between user_embeddings and negative_embeddings as negative logits [BxN, num_negatives]
    # 4. softmax(pos_logits, neg_logits)
    def forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            output_embeddings: [B, N, D] x float, embeddings for the current
                input sequence.
            supervision_ids: [B, N] x int64, (positive) supervision ids.
            supervision_embeddings: [B, N, D] x float.

        Returns:
            torch.Tensor: loss for the current engaged sequence.
        """
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.numel() == supervision_embeddings.size(0)
        (
            sampled_candidate_ids,
            sampled_candidate_embeddings,
        ) = self._negatives_sampler.gen_sampled_candidates(
            supervision_ids, supervision_embeddings
        )
        sampled_ids, sampled_negative_embeddings = self._negatives_sampler(
            positive_ids=supervision_ids,
            num_to_sample=self._num_to_sample
            if self._num_to_sample > 0
            else supervision_ids.numel(),
            sampled_candidate_ids=sampled_candidate_ids,
            sampled_candidate_embeddings=sampled_candidate_embeddings,
        )
        positive_embeddings = self._negatives_sampler.normalize_embeddings(
            supervision_embeddings
        )
        positive_logits = (
            self._interaction_module(
                input_embeddings=output_embeddings,  # [B, D] = [N', D]
                item_embeddings=positive_embeddings.unsqueeze(
                    1
                ),  # [N', D] -> [N', 1, D]
            )
            / self._softmax_temperature
        )
        sampled_negatives_logits = self._interaction_module(
            input_embeddings=output_embeddings,  # [N', D]
            item_embeddings=sampled_negative_embeddings,  # [N', R, D]
        )
        sampled_negatives_logits = torch.where(
            supervision_ids.unsqueeze(1) == sampled_ids,  # [N', R]
            -5e4,
            sampled_negatives_logits / self._softmax_temperature,
        )
        losses = -F.log_softmax(
            torch.cat([positive_logits, sampled_negatives_logits], dim=1), dim=1
        )[:, 0]
        return losses
