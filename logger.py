
from typing import Callable, Dict
import time
from datetime import datetime
from rich.progress import Progress, TaskID


class Logger:
    def __init__(self, num_epochs: int, description: str = "Training", 
                 completed: int =0):
        self.num_epochs = num_epochs
        self.description = description
        self.progress_bar = None
        self.epoch_task = None
        self.completed = completed;

    def __enter__(self):
        self.progress_bar = Progress()
        self.epoch_task: TaskID = self.progress_bar.add_task(f"[cyan]{self.description}...", 
                                                             completed = self.completed,
                                                             total=self.num_epochs)
        self.progress_bar.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress_bar:
            self.progress_bar.stop()
        self.progress_bar = None
        self.epoch_task = None

    def epoch_callback(self, trainer):

        if self.progress_bar and self.epoch_task is not None:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_train_loss = trainer.current_train_loss if hasattr(trainer, 'current_train_loss') else 0.0
            current_test_loss = trainer.current_test_loss if hasattr(trainer, 'current_test_loss') else 0.0
            self.progress_bar.update(
                self.epoch_task,
                advance=1,
                description=f"Epoch: {trainer.current_epoch} Timestep {trainer.current_time_step} @ {current_time}, test: {current_test_loss: .5f}, train: {current_train_loss: .5f}"
            )

    def training_complete_callback(self, trainer):
        return
        if self.progress_bar and self.epoch_task is not None:
            self.progress_bar.remove_task(self.epoch_task)
            print(f"\nTraining complete for {self.description}.")
