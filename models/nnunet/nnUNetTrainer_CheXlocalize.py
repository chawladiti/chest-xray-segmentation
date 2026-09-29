"""Custom nnU-Net v2 trainer for the CheXlocalize full experiment.

Honours the project's training contract via environment variables (so LR / batch /
epochs / patience are configurable without editing code):

    NNUNET_NUM_EPOCHS   (default 20)   -> maximum epochs (project cap = 20)
    NNUNET_LR           (default nnU-Net default 1e-2) -> initial learning rate
    NNUNET_BATCH        (default from plans) -> batch size override
    NNUNET_PATIENCE     (default 0 = off)  -> early-stopping patience in epochs,
                        measured on nnU-Net's EMA pseudo-Dice (validation metric).

Native nnU-Net behaviour that already satisfies the contract:
  * validation / progress every epoch (progress.png, training_log, summary.json)
  * best checkpoint  -> checkpoint_best.pth (tracks best EMA pseudo-Dice)
  * resume training  -> `nnUNetv2_train ... --c` (continues from checkpoint_latest)

To make nnU-Net discover this trainer, copy it into the installed package:
    models/nnunet/install_trainer.sh
Then train with:  nnUNetv2_train <DID> 2d <FOLD> -tr nnUNetTrainer_CheXlocalize
"""

from __future__ import annotations

import os

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_CheXlocalize(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # --- configurable training contract ---------------------------------
        self.num_epochs = int(os.environ.get("NNUNET_NUM_EPOCHS", 20))
        self.initial_lr = float(os.environ.get("NNUNET_LR", self.initial_lr))

        batch = os.environ.get("NNUNET_BATCH")
        if batch:
            # configuration_manager.batch_size reads from this dict
            self.configuration_manager.configuration["batch_size"] = int(batch)

        # --- optional early stopping (patience on EMA pseudo-Dice) ----------
        self._patience = int(os.environ.get("NNUNET_PATIENCE", 0))
        self._best_ema = None
        self._epochs_without_improve = 0

        self.print_to_log_file(
            f"[CheXlocalize] num_epochs={self.num_epochs} initial_lr={self.initial_lr} "
            f"batch_size={self.configuration_manager.batch_size} "
            f"early_stop_patience={self._patience}"
        )

    def on_epoch_end(self):
        super().on_epoch_end()
        if self._patience <= 0:
            return
        ema = getattr(self, "_best_ema_pseudo_dice", None)
        current = self.logger.my_fantastic_logging["ema_fg_dice"][-1] \
            if self.logger.my_fantastic_logging.get("ema_fg_dice") else None
        if current is None:
            return
        if self._best_ema is None or current > self._best_ema + 1e-4:
            self._best_ema = current
            self._epochs_without_improve = 0
        else:
            self._epochs_without_improve += 1
            if self._epochs_without_improve >= self._patience:
                self.print_to_log_file(
                    f"[CheXlocalize] early stopping: no EMA-Dice improvement for "
                    f"{self._patience} epochs (best={self._best_ema:.4f}). "
                    f"Ending at epoch {self.current_epoch}."
                )
                # end the training loop gracefully after this epoch
                self.num_epochs = self.current_epoch + 1
