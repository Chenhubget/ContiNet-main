from Utils.Packages import *

# Define custom metrics
def R2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
def MAE(y_true, y_pred): return  K.mean(K.abs( y_true - y_pred ))
def SAE(y_true, y_pred): return  K.sum(K.abs( y_true - y_pred ))  
def MSE(y_true, y_pred): return  K.mean(K.square( y_true - y_pred ))
def MAPE(y_true, y_pred): return K.mean(K.abs( (y_true - y_pred )/y_true))

# Define custom callback
class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}/{self.params['epochs']}: loss: {logs['loss']}, R2: {logs['R2']}, MAE: {logs['MAE']}")
            
