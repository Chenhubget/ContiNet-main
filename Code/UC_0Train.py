from Utils.Packages import *
from Utils.Auxiliary import *
from Utils.ContiNet_model import *

# # File paths and Load data
# parent_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)).replace('\\', '/')
# ResAnalyFile = parent_directory +  +'/Results_1DiscreteUC_1DataSet_Evaluation/'   # Results analysis file path
# UTrainSet = 'UTrainSet4'  #UTrainSet can be 'UTrainSet1' 'UTrainSet2' 'UTrainSet3' 'UTrainSet4' etc.
# DataFile = parent_directory + '/DataSet_UC/'+ UTrainSet +'/'  # Data file path
# Data_low = np.array(h5py.File(DataFile + 'Data_ZLow_1to40.mat')['Data_ZLow']).T  
# Cond_upheight = np.array(h5py.File(DataFile + 'Cond_upheight_1to40.mat')['Cond_upheight']).T
# Cond_upkernel = np.array(h5py.File(DataFile + 'Cond_upkernel_1to40.mat')['Cond_upkernel']).T
# Strr = 'Discrete'  # 'Discrete' or 'Theoretical'
# if Strr == 'Discrete':
#     DataSet_ZUp = np.array(h5py.File(DataFile + 'Data_ZUp_1to40.mat')['Data_ZUp']).T
#     Data_high = DataSet_ZUp
#     DataSet_Zhigh = np.array(h5py.File(DataFile + 'Data_ZHigh_1to40.mat')['Data_ZHigh']).T
#     DataSet_Zhigh = np.array(h5py.File(DataFile + 'Data_ZHigh_1to40.mat')['Data_ZHigh']).T
#     Data_high = DataSet_Zhigh

# # Model training
# Layers = 9   # Layers can be modified when testing different layers of the model
# Calnum = "1"    # Number of calculation points 1 or all
# if Strr == 'Discrete':
#     model = UpContiNet_origin(Data_low, Data_high[:,250:251], Cond_upheight)
# if Strr == 'Theoretical':
#     if Calnum == 1:
#         model = UpContiNet_sp(Data_low, Data_high[:,250:251], Cond_upheight)
#     else:
#         model = UpContiNet_ap(Data_low, Data_high, Cond_upheight)
# model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=1e-6),metrics=[R2,MAE,MSE,MAPE])    #The Lr can be adjusted to achieve better results
# model.summary()
# checkpoint = ModelCheckpoint(ResAnalyFile +'/UC_'+ Strr +'_Modelparam_'+ UTrainSet +'_Model'+'_Layers'+str(Layers)+'_CalPNum'+Calnum+'.h5', monitor='loss', save_best_only=True)
# log_dir = ResAnalyFile +'/UC_'+ Strr +'_Trainlogs_'+ UTrainSet +'_Model'+'_Layers' + str(Layers) + '_CalPNum'+Calnum + '_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# callbackss = [tensorboard_callback,checkpoint,CustomCallback()]   
# history = model.fit([Data_low,Cond_upheight], Data_high, epochs=1, batch_size=501, shuffle=True, verbose=0, callbacks=callbackss) # The batch_size/Epochs can be adjusted to achieve better results
# history_dict = history.history
# with open(ResAnalyFile +'/UC_'+ Strr +'_History_'+ UTrainSet +'_Model'+'_Layers' + str(Layers) + '_CalPNum'+Calnum + '.json', 'w') as f:
#         json.dump(history_dict, f)
# MinIndex=np.argmin(history.history['loss'])
# print('MinIndex:',MinIndex,'loss:',history.history['loss'][MinIndex],'R2:',history.history['R2'][MinIndex],'MAE:',history.history['MAE'][MinIndex],'MAPE:',history.history['MAPE'][MinIndex])

# # Training results
# model = load_model(ResAnalyFile +'/UC_'+ Strr +'_Modelparam_'+ UTrainSet +'_Model'+'_Layers'+str(Layers)+'_CalPNum'+Calnum+'.h5',custom_objects={'R2':R2})
# Midfiture_label = Model(inputs=model.input,outputs=model.get_layer('dense').output)   # get_layer(.)  where . can be replaced by the name of the layer,such as 'dense_8','lambda_1' etc.
# Upkernel_pred = Midfiture_label.predict([Data_low[14:14+14,:],Cond_upheight[14:14+14,:]])
# Upkernel_pred=Upkernel_pred.tolist()
# with open(ResAnalyFile +'/UC_'+ Strr +'_PredUCkernel_'+ UTrainSet +'_Model'+'_Layers' + str(Layers) + '_CalPNum'+Calnum + '.json', 'w') as f:
#         json.dump(Upkernel_pred, f)