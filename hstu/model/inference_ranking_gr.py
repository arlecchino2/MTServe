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
import os

import torch
from commons.datasets.hstu_batch import HSTUBatch
from configs import InferenceHSTUConfig, KVCacheConfig, RankingConfig
from modules.inference_dense_module import InferenceDenseModule
from modules.inference_embedding import InferenceEmbedding

import time


class InferenceRankingGR(torch.nn.Module):
    """
    A class representing the ranking model inference.

    Args:
        sparse_module (InferenceHSTUConfig): The HSTU configuration.
        dense_module (RankingConfig): The ranking task configuration.
    """

    def __init__(
        self,
        sparse_module: torch.nn.Module,
        dense_module: torch.nn.Module,
        enable_timing_stats=False,
        logger=None,
    ):
        super().__init__()
        self.sparse_module = sparse_module
        self.dense_module = dense_module
        
        self.logger = logger
        self._device = torch.cuda.current_device()
        
        self.enable_timing_stats = enable_timing_stats
        if self.enable_timing_stats:
            self._init_timing_stats()

    def _init_timing_stats(self):
        self.timing_stats = {
            'forward_with_cache': {'count': 0, 'total_time': 0.0, 'step_times': {}},
            'forward_no_cache': {'count': 0, 'total_time': 0.0, 'step_times': {}}
        }
        self.count = 0
        self.cache_stats = {
            'total_gpu_length': 0,
            'total_host_load_length': 0,
            'total_new_tokens': 0,
            'total_sequence_length': 0,
            'batch_count': 0
        }

    def _update_timing_stats(self, method_name, timing_info):
        if not self.enable_timing_stats:
            return

        stats = self.timing_stats.setdefault(
            method_name,
            {'count': 0, 'total_time': 0.0, 'step_times': {}}
        )
        stats['count'] += 1
        stats['total_time'] += timing_info.get('total_time', 0.0)

        for step, duration in timing_info.items():
            if step == 'total_time':
                continue
            if step not in stats['step_times']:
                stats['step_times'][step] = {'total': 0.0, 'count': 0}
            stats['step_times'][step]['total'] += duration
            stats['step_times'][step]['count'] += 1

    def _print_timing_summary(self):
        if not self.enable_timing_stats:
            return

        def _log(message):
            if self.logger is not None:
                self.logger.info(message)
            print(message)

        _log("=" * 60)
        _log("TIMING SUMMARY")
        _log("=" * 60)

        for method_name, stats in self.timing_stats.items():
            if stats['count'] <= 0:
                continue

            avg_total = stats['total_time'] / stats['count']
            _log(f"\n{method_name.upper()}:")
            _log(f"  Total calls: {stats['count']}")
            _log(f"  Total time: {stats['total_time']:.6f}s")
            _log(f"  Average time per call: {avg_total:.6f}s")
            _log("  Step-wise averages:")

            for step, step_stats in stats['step_times'].items():
                if step_stats['count'] == 0:
                    continue
                avg_step = step_stats['total'] / step_stats['count']
                percentage = (avg_step / avg_total * 100) if avg_total else 0.0
                _log(f"    {step}: {avg_step:.6f}s ({percentage:.1f}%)")

    def analyze_cache_distribution(
        self,
        user_ids: torch.Tensor,
        total_history_lengths: torch.Tensor,
    ):
        with torch.inference_mode():
            gpu_cache_info = self.dense_module.async_kvcache.gpu_kvcache_mgr.get_cache_startpos_and_length(
                user_ids.tolist()
            )
            host_cache_info = self.dense_module.async_kvcache.host_kv_mgr.get_cache_startpos_and_length(
                user_ids.tolist()
            )

            batch_size = len(user_ids)
            cache_distribution = {
                'user_ids': user_ids.tolist(),
                'total_lengths': total_history_lengths.tolist(),
                'gpu_cache': [],
                'host_cache': [],
                'new_tokens': []
            }

            for i in range(batch_size):
                total_length = total_history_lengths[i].item()
                gpu_start, gpu_length = gpu_cache_info[i]
                host_start, host_length = host_cache_info[i]
                host_load_length = gpu_start
                gpu_cache_length = gpu_length
                new_tokens = total_length - (gpu_start + gpu_length)

                cache_distribution['gpu_cache'].append({
                    'start': gpu_start,
                    'length': gpu_cache_length
                })
                cache_distribution['host_cache'].append({
                    'start': host_start,
                    'length': host_length,
                    'load_length': host_load_length
                })
                cache_distribution['new_tokens'].append(new_tokens)

            return cache_distribution

    def update_cache_stats(self, cache_distribution):
        if not self.enable_timing_stats:
            return

        self.count += 1
        batch_gpu_length = sum(gpu['length'] for gpu in cache_distribution['gpu_cache'])
        batch_host_load_length = sum(host['load_length'] for host in cache_distribution['host_cache'])
        batch_new_tokens = sum(cache_distribution['new_tokens'])
        batch_total_length = sum(cache_distribution['total_lengths'])

        if self.count > 2999:
            self.cache_stats['total_gpu_length'] += batch_gpu_length
            self.cache_stats['total_host_load_length'] += batch_host_load_length
            self.cache_stats['total_new_tokens'] += batch_new_tokens
            self.cache_stats['total_sequence_length'] += batch_total_length
            self.cache_stats['batch_count'] += 1

    def print_cache_summary(self):
        if not self.enable_timing_stats or self.cache_stats['batch_count'] == 0:
            return

        total_sequence_length = self.cache_stats['total_sequence_length']
        if total_sequence_length == 0:
            return

        total_cached = (
            self.cache_stats['total_gpu_length'] + self.cache_stats['total_host_load_length']
        )
        gpu_hit_rate = self.cache_stats['total_gpu_length'] / total_sequence_length * 100
        host_hit_rate = self.cache_stats['total_host_load_length'] / total_sequence_length * 100
        new_token_rate = self.cache_stats['total_new_tokens'] / total_sequence_length * 100

        def _log(message):
            if self.logger is not None:
                self.logger.info(message)
            print(message)

        _log("=" * 60)
        _log("CACHE HIT RATE SUMMARY")
        _log("=" * 60)
        _log(f"Total Batches: {self.cache_stats['batch_count']}")
        _log(f"Total Sequence Length: {total_sequence_length}")
        _log(
            f"Total Host Load Length: {self.cache_stats['total_host_load_length']} ({host_hit_rate:.2f}%)"
        )
        _log(
            f"Total GPU Cache Length: {self.cache_stats['total_gpu_length']} ({gpu_hit_rate:.2f}%)"
        )
        _log(
            f"Total New Tokens: {self.cache_stats['total_new_tokens']} ({new_token_rate:.2f}%)"
        )
        _log(
            f"Total Cached: {total_cached} ({(total_cached / total_sequence_length * 100):.2f}%)"
        )
        _log(f"GPU Cache Hit Rate: {gpu_hit_rate:.2f}%")
        _log("=" * 60)

    def bfloat16(self):
        """
        Convert the model to use bfloat16 precision. Only affects the dense module.

        Returns:
            RankingGR: The model with bfloat16 precision.
        """
        self.dense_module.bfloat16()
        return self

    def half(self):
        """
        Convert the model to use half precision. Only affects the dense module.

        Returns:
            RankingGR: The model with half precision.
        """
        self.dense_module.half()
        return self

    def get_num_class(self):
        return self.dense_module.get_num_class()

    def get_num_tasks(self):
        return self.dense_module.get_num_tasks()

    def get_metric_types(self):
        return self.dense_module.get_metric_types()

    def load_checkpoint(self, checkpoint_dir):
        if checkpoint_dir is None:
            return

        model_state_dict_path = os.path.join(
            checkpoint_dir, "torch_module", "model.0.pth"
        )
        model_state_dict = torch.load(model_state_dict_path)["model_state_dict"]

        self.sparse_module.load_checkpoint(checkpoint_dir, model_state_dict)
        self.dense_module.load_state_dict(model_state_dict, strict=False)

    def forward_with_kvcache(
        self,
        batch: HSTUBatch,
        user_ids: torch.Tensor,
        total_history_lengths: torch.Tensor,
    ):
        with torch.inference_mode():
            cache_distribution = self.analyze_cache_distribution(user_ids, total_history_lengths)
            self.update_cache_stats(cache_distribution)
            
            # 1. 准备KV Cache
            if self.enable_timing_stats:
                start_time = time.time()
                timing_info = {}
                prepare_kvcache_start = time.time()
            
            prepare_kvcache_result = (
                self.dense_module.async_kvcache.prepare_kvcache_async(
                    batch.batch_size,
                    user_ids.tolist(),
                    total_history_lengths.tolist(),
                    self.dense_module.async_kvcache.static_page_ids_gpu_buffer,
                    self.dense_module.async_kvcache.static_offload_page_ids_gpu_buffer,
                    self.dense_module.async_kvcache.static_metadata_gpu_buffer,
                    self.dense_module.async_kvcache.static_onload_handle,
                )
            )

            old_cached_lengths = torch.tensor(
                prepare_kvcache_result[0], dtype=torch.int32
            )
            
            if self.enable_timing_stats:
                # torch.cuda.synchronize()
                timing_info['prepare_kvcache_async'] = time.time() - prepare_kvcache_start
            
            # 2. 去除已缓存的token
            if self.enable_timing_stats:
                strip_cached_start = time.time()
            striped_batch = self.dense_module.async_kvcache.strip_cached_tokens(
                batch,
                old_cached_lengths,
            )
            if self.enable_timing_stats:
                # torch.cuda.synchronize()
                timing_info['strip_cached_tokens'] = time.time() - strip_cached_start

            # 3. Embedding计算
            torch.cuda.nvtx.range_push("HSTU embedding")
            if self.enable_timing_stats:
                emb_start = time.time()
            embeddings = self.sparse_module(striped_batch.features)
            torch.cuda.nvtx.range_pop()
            if self.enable_timing_stats:
                # torch.cuda.synchronize()
                timing_info['embedding'] = time.time() - emb_start

            prepare_kvcache_result = [old_cached_lengths] + prepare_kvcache_result[1:]
            logits = self.dense_module.forward_with_kvcache(
                striped_batch,
                embeddings,
                user_ids,
                total_history_lengths,
                prepare_kvcache_result,
                timing_info if self.enable_timing_stats else None,
            )
            if self.enable_timing_stats:
                timing_info['total_time'] = time.time() - start_time
                self.count += 1
                if self.count > 3000:
                    self._update_timing_stats('forward_with_cache', timing_info)
        return logits

    def forward_nokvcache(
        self,
        batch: HSTUBatch,
    ):
        with torch.inference_mode():
            torch.cuda.nvtx.range_push("HSTU embedding")
            embeddings = self.sparse_module(batch.features)
            torch.cuda.nvtx.range_pop()
            logits = self.dense_module.forward_nokvcache(batch, embeddings)

        return logits

    def forward(
        self,
        batch: HSTUBatch,
    ):
        with torch.inference_mode():
            torch.cuda.nvtx.range_push("HSTU embedding")
            embeddings = self.sparse_module(batch.features)
            torch.cuda.nvtx.range_pop()
            logits = self.dense_module(batch, embeddings)
        return logits


def get_inference_ranking_gr(
    hstu_config: InferenceHSTUConfig,
    kvcache_config: KVCacheConfig,
    task_config: RankingConfig,
    use_cudagraph=False,
    enable_timing_stats=False,
    logger=None,
    cudagraph_configs=None,
    sparse_shareables=None,
):
    for ebc_config in task_config.embedding_configs:
        assert (
            ebc_config.dim == hstu_config.hidden_size
        ), "hstu layer hidden size should equal to embedding dim"

    inference_sparse = InferenceEmbedding(
        task_config.embedding_configs,
        sparse_shareables,
    )
    inference_dense = InferenceDenseModule(
        hstu_config,
        kvcache_config,
        task_config,
        use_cudagraph,
        cudagraph_configs,
        enable_timing_stats,
        logger
    )

    return InferenceRankingGR(inference_sparse, inference_dense, enable_timing_stats, logger)
