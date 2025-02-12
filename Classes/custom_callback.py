from tensorflow.keras.callbacks import Callback

class CustomCallback(Callback):

    
    def on_epoch_end(self,epoch,logs = None):
        print(f"Epoch {epoch +1}: Custom callback log message")
