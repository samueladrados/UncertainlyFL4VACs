import tensorflow as tf
import numpy as np

class ValidationCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to run specific validations at the end of each epoch.
    In a Federated Learning environment, we keep this lightweight to avoid 
    slowing down the training process on the clients.
    """

    def __init__(self, val_gen, log_dir=None):
        """
        Args:
            val_gen: The validation data generator.
            log_dir: Directory to save logs (optional).
        """
        super(ValidationCallback, self).__init__()
        self.val_gen = val_gen
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        """
        Executed at the end of each epoch.
        """
        if logs is None:
            logs = {}
            
        # Get the current training loss
        loss = logs.get('loss', 0.0)
        
        # Simple message to confirm the client is still alive and training
        print(f"    📉 [Client Callback] Epoch {epoch+1} finished. Loss: {loss:.4f}")
