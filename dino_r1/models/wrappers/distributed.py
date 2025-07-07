# Copyright (c) Robert Bosch LLC CR/RHI1-NA. All rights reserved.
import copy
from typing import Any, Dict, Union

import torch

from mmengine.optim import OptimWrapper
from mmengine.registry import MODEL_WRAPPERS
from mmengine.model.wrappers import MMDistributedDataParallel
from mmengine.model.utils import detect_anomalous_params

@MODEL_WRAPPERS.register_module()
class MMDistributedDataParallel_DINO_R1(MMDistributedDataParallel):
    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.module.data_preprocessor(data, training=True)
            losses = self._run_forward(data, mode='loss')
        parsed_loss, log_vars = self.module.parse_losses(losses)
        optim_wrapper.update_params(parsed_loss)
        if self.detect_anomalous_params:
            detect_anomalous_params(parsed_loss, model=self)
        return log_vars
    
    def forward(self, *inputs, **kwargs):
        with torch.autograd.profiler.record_function("DistributedDataParallel.forward"):
            inputs, kwargs = self._pre_forward(*inputs, **kwargs)

            with self._inside_ddp_forward():
                ref_kwargs = copy.deepcopy(kwargs)
                ref_kwargs['mode'] = 'grpo'
                ref_select_score_per_token = self.ref_module(*inputs, **ref_kwargs) 
                kwargs['data_samples'][0].ref_select_score_per_token = ref_select_score_per_token 

            output, _ = (
                self.module.forward(*inputs, **kwargs)
                if self._delay_all_reduce_all_params
                else self._run_ddp_forward(*inputs, **kwargs)
            ) 
            return self._post_forward(output)


