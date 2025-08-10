from Utils.Packages import *
from Utils.Auxiliary import *
from Utils.ContiNet_model import *

# # Load Data
# parent_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)).replace('\\', '/')
# DataFile = parent_directory + '/DataSet_DC/'    
# ResAnalyFile= parent_directory + '/Results_DC/Theoretical_1TrainingResults/'
# Data_low1 = np.array(h5py.File(DataFile + 'Data_ZLow_fuhe.mat')['Data_ZLow']).T
# Data_high1 = np.array(h5py.File(DataFile + 'Data_ZHigh_fuhe.mat')['Data_ZHigh']).T
# Cond_downheight1 = np.array(h5py.File(DataFile + 'Cond_Downheight_fuhe.mat')['Cond_Downheight']).T   
# Data_low2 = np.array(h5py.File(DataFile + 'Data_ZLow_single.mat')['Data_ZLow']).T
# Data_high2 = np.array(h5py.File(DataFile + 'Data_ZHigh_single.mat')['Data_ZHigh']).T
# Cond_downheight2 = np.array(h5py.File(DataFile + 'Cond_Downheight_single.mat')['Cond_Downheight']).T
# Data_low = np.concatenate((Data_low1[0:14,:],Data_low2[0:14,:]),axis=0)
# Data_high = np.concatenate((Data_high1[0:14,:],Data_high2[0:14,:]),axis=0)
# Cond_downheight = np.concatenate((Cond_downheight1[0:14,:],Cond_downheight2[0:14,:]),axis=0)
# Cond_upkernel = np.array(h5py.File(DataFile + 'Cond_upkernel.mat')['Cond_upkernel']).T

# # Normalization
# Data_high_normed = (Data_high - np.mean(Data_high,axis=0)) / np.std(Data_high,axis=0)
# Data_low_normed = (Data_low - np.mean(Data_low,axis=0)) / np.std(Data_low,axis=0)
# label_embedding = Cond_downheight / np.max(Cond_downheight,axis=0)  

# # Model training
# Strr='With' # 'With' for regularization, 'No' for no regularization
# model = DnContiNet(Data_high_normed,Data_low_normed,label_embedding)   
# model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),metrics=[R2,MAE,MSE,MAPE])
# model.summary()
# checkpoint = ModelCheckpoint(ResAnalyFile + 'DC_Theoretical_Modelparam_DTrainSet1and2_'+ Strr+'Reg_Model_Layers9_CalPNumall.h5', monitor='loss', save_best_only=True)
# log_dir = ResAnalyFile + 'DC_Theoretical_Trainlogs_DTrainSet1and2_'+ Strr + 'Reg_Model_Layers9_CalPNumall_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# callbackss = [tensorboard_callback,checkpoint,CustomCallback()]
# history = model.fit([Data_high_normed ,label_embedding], Data_low_normed, epochs=1, batch_size=501, shuffle=True, verbose=0, callbacks=callbackss)  
# history_dict = history.history
# with open(ResAnalyFile + 'DC_Theoretical_History_DTrainSet1and2_'+ Strr + 'Reg_Model_Layers9_CalPNumall_lr1ef4_epoch500_batch501.json', 'w') as f:
#         json.dump(history_dict, f)
# MinIndex=np.argmin(history.history['loss'])
# print('MinIndex:',MinIndex,'loss:',history.history['loss'][MinIndex],'R2:',history.history['R2'][MinIndex],'MAE:',history.history['MAE'][MinIndex],'MAPE:',history.history['MAPE'][MinIndex])

