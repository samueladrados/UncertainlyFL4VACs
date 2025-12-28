import os
import csv
import tensorflow as tf

class FederatedLogger(tf.keras.callbacks.Callback):
    def __init__(self, client_id, round_num, log_dir="fl_logs"):
        super(FederatedLogger, self).__init__()
        self.client_id = client_id
        self.round_num = round_num
        self.log_dir = log_dir
        
        # Crear directorio si no existe
        os.makedirs(log_dir, exist_ok=True)
        
        # Archivo específico para este cliente
        self.file_path = os.path.join(log_dir, f"{client_id}_history.csv")
        
        # Definir las columnas que queremos guardar
        self.fieldnames = ['round', 'epoch', 'batch', 'loss', 'cls_loss', 'box_loss', 'kl_loss']
        
        # Si el archivo no existe, crearlo y poner cabeceras
        if not os.path.exists(self.file_path):
            with open(self.file_path, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def on_epoch_end(self, epoch, logs=None):
        """Se ejecuta al final de cada época."""
        logs = logs or {}
        
        # Preparar fila de datos
        row = {
            'round': self.round_num,
            'epoch': epoch + 1,      # Epoch 1-based
            'batch': 'End-Epoch',    # Marcador de final de época
            'loss': f"{logs.get('loss', 0):.4f}",
            # Ajusta estas claves según los nombres reales en tu model.metrics_names
            # Normalmente son output_1, output_2... o los nombres que definiste en compute_loss
            'cls_loss': f"{logs.get('cls_loss', 0):.4f}", 
            'box_loss': f"{logs.get('box_loss', 0):.4f}",
            'kl_loss':  f"{logs.get('kl_loss', 0):.4f}"
        }
        
        self._write_row(row)

    def on_train_batch_end(self, batch, logs=None):
        """(OPCIONAL) Se ejecuta al final de cada batch. 
           Descomenta si quieres ultra-detalle, pero genera archivos gigantes."""
        # logs = logs or {}
        # row = {
        #     'round': self.round_num,
        #     'epoch': 'Current',
        #     'batch': batch,
        #     'loss': f"{logs.get('loss', 0):.4f}",
        #     'cls_loss': f"{logs.get('cls_loss', 0):.4f}",
        #     'box_loss': f"{logs.get('box_loss', 0):.4f}",
        #     'kl_loss':  f"{logs.get('kl_loss', 0):.4f}"
        # }
        # self._write_row(row)
        pass

    def _write_row(self, row):
        """Escribe una fila en el CSV de forma segura."""
        with open(self.file_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)