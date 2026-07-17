import numpy as np


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-4,
                 verbose: bool = True):
        """
        Args:
            patience: How many epochs to wait after last time validation loss improved.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
            verbose: If True, prints a message for each validation loss improvement.
        """
        self.patience  = patience
        self.min_delta = min_delta
        self.verbose   = verbose

        self.best_loss       = np.inf
        self.counter         = 0
        self.early_stop      = False
        self.save_checkpoint = False

    def __call__(self, val_loss: float) -> None:
        """
        Call this method at the end of each epoch to check if training should stop.
        
        Args:
            val_loss: The validation loss of the current epoch.
        """
        self.save_checkpoint = False

        if val_loss < (self.best_loss - self.min_delta):
            if self.verbose:
                print(f"    Improvement: {self.best_loss:.6f} → {val_loss:.6f}")
            self.best_loss       = val_loss
            self.counter         = 0
            self.save_checkpoint = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
