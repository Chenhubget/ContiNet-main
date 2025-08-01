from Utils.Packages import *
from Utils.Auxiliary import *
from Utils.ContiNet_model import *

# Load data
parent_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)).replace('\\', '/')
UTrainSet = 'UTrainSet1' # UTrainSet can be 'UTrainSet1' 'UTrainSet2' 'UTrainSet3' 'UTrainSet4' etc.
DataFile = parent_directory + '/Data/DataSet_UC/'+ UTrainSet +'/'
ResAnalyFile = parent_directory + '/Output/Results_UC/'+ UTrainSet +'/'
DataSet_ZLow = np.array(h5py.File(DataFile + 'Data_ZLow.mat')['Data_ZLow']).T
DataSet_ZUp = np.array(h5py.File(DataFile + 'Data_ZUp.mat')['Data_ZUp']).T
Data_low = DataSet_ZLow
Data_high = DataSet_ZUp[:,250:251]
Cond_upheight = np.array(h5py.File(DataFile + 'Cond_upheight.mat')['Cond_upheight']).T
Cond_upkernel = np.array(h5py.File(DataFile + 'Cond_upkernel.mat')['Cond_upkernel']).T

# Model training
i=9   # i can be 1 3 9 when testing different layers of the model
# model =UpContiNet_origin(Data_low,Data_high,Cond_upheight)
# model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=1e-6),metrics=[R2,MAE,MSE,MAPE])    
# model.summary()
# checkpoint = ModelCheckpoint(ResAnalyFile +'/best_model_'+str(i)+'.h5', monitor='loss', save_best_only=True)
# log_dir = ResAnalyFile + "/trainlogs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# callbackss = [tensorboard_callback,checkpoint,CustomCallback()]   
# history = model.fit([Data_low,Cond_upheight], Data_high, epochs=1, batch_size=501, shuffle=True, verbose=0, callbacks=callbackss) 
# history_dict = history.history
# with open(ResAnalyFile + '/history_'+str(i)+'.json', 'w') as f:
#         json.dump(history_dict, f)
# MinIndex=np.argmin(history.history['loss'])
# print('MinIndex:',MinIndex,'loss:',history.history['loss'][MinIndex],'R2:',history.history['R2'][MinIndex],'MAE:',history.history['MAE'][MinIndex],'MAPE:',history.history['MAPE'][MinIndex])

# # Training result 
# model = load_model(ResAnalyFile + '/best_model_'+str(i)+'.h5',custom_objects={'R2':R2})
# Midfiture_label = Model(inputs=model.input,outputs=model.get_layer('dense').output)
# Upkernel_pred = Midfiture_label.predict([Data_low[14:14+14,:],Cond_upheight[14:14+14,:]])
# Upkernel_pred=Upkernel_pred.tolist()
# with open(ResAnalyFile + '/Upkernel_pred_'+str(i)+'.json', 'w') as f:
#         json.dump(Upkernel_pred, f)