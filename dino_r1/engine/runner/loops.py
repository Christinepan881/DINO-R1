# Copyright (c) Robert Bosch LLC CR/RHI1-NA. All rights reserved.
import copy
from typing import Sequence
from mmdet.registry import LOOPS
from mmengine.runner import EpochBasedTrainLoop

@LOOPS.register_module()
class EpochBasedTrainLoop_DINO_R1(EpochBasedTrainLoop):

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()

        self.runner.model.ref_module = copy.deepcopy(self.runner.model.module)

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook('before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)

        self._iter += 1


