from Utils.Packages import *
from Utils.Auxiliary import *
from Utils.ContiNet_model import *

## Load data
parent_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)).replace('\\', '/')
ResAnalyFile= parent_directory +'/Output/Results_UC/Theoretical_weight/Allpoint/'
# 1to40
DataFile_1to40 = parent_directory +'/Data/DataSet_UC/UTrainSet4/'    
DataSet_ZLow_1to40 = np.array(h5py.File(DataFile_1to40 + 'Data_ZLow.mat')['Data_ZLow']).T
DataSet_ZHigh_1to40 = np.array(h5py.File(DataFile_1to40 + 'Data_ZHigh.mat')['Data_ZHigh']).T
Data_low_1to40 = DataSet_ZLow_1to40[:,:]
Data_high_1to40 = DataSet_ZHigh_1to40   
Cond_upheight_1to40 = np.array(h5py.File(DataFile_1to40 + 'Cond_upheight.mat')['Cond_upheight']).T
Cond_upkernel_1to40 = np.array(h5py.File(DataFile_1to40 + 'Cond_upkernel.mat')['Cond_upkernel']).T
# 2to38
DataFile_test = parent_directory + '/Data/DataSet_UC/GeneralizationSet/' 
DataSet_ZLow_2to38 = np.array(h5py.File(DataFile_test + 'Data_ZLow_2to38.mat')['Data_ZLow']).T
DataSet_ZHigh_2to38 = np.array(h5py.File(DataFile_test + 'Data_ZHigh_2to38.mat')['Data_ZHigh']).T
Data_low_2to38 = DataSet_ZLow_2to38[:,:]
Data_high_2to38 = DataSet_ZHigh_2to38   
Cond_upheight_2to38 = np.array(h5py.File(DataFile_test + 'Cond_upheight_2to38.mat')['Cond_upheight']).T
Cond_upkernel_2to38 = np.array(h5py.File(DataFile_test + 'Cond_upkernel_2to38.mat')['Cond_upkernel']).T
# 3to39
DataSet_ZLow_3to39 = np.array(h5py.File(DataFile_test + 'Data_ZLow_3to39.mat')['Data_ZLow']).T
DataSet_ZHigh_3to39 = np.array(h5py.File(DataFile_test + 'Data_ZHigh_3to39.mat')['Data_ZHigh']).T
Data_low_3to39 = DataSet_ZLow_3to39[:,:]
Data_high_3to39 = DataSet_ZHigh_3to39   
Cond_upheight_3to39 = np.array(h5py.File(DataFile_test + 'Cond_upheight_3to39.mat')['Cond_upheight']).T
Cond_upkernel_3to39 = np.array(h5py.File(DataFile_test + 'Cond_upkernel_3to39.mat')['Cond_upkernel']).T
# Concatenate Data
DataSet_ZLow = np.vstack((DataSet_ZLow_1to40,DataSet_ZLow_2to38,DataSet_ZLow_3to39))
DataSet_ZHigh = np.vstack((DataSet_ZHigh_1to40,DataSet_ZHigh_2to38,DataSet_ZHigh_3to39))
Data_low = DataSet_ZLow[:,:]
Data_high = DataSet_ZHigh   
Cond_upheight = np.vstack((Cond_upheight_1to40,Cond_upheight_2to38,Cond_upheight_3to39))
Cond_upkernel = np.vstack((Cond_upkernel_1to40,Cond_upkernel_2to38,Cond_upkernel_3to39))

# Model training
# Strr='_theoretical'
# model =UpContiNet_ap(Data_low,Data_high,Cond_upheight)  
# model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=1e-6),metrics=[R2,MAE,MSE,MAPE])    
# model.summary()
# checkpoint = ModelCheckpoint(ResAnalyFile +'/best_model_'+str(9)+Strr+'.h5', monitor='loss', save_best_only=True)
# log_dir = ResAnalyFile + "/trainlogs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+Strr
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# callbackss = [tensorboard_callback,checkpoint,CustomCallback()]   
# history = model.fit([Data_low,Cond_upheight], Data_high, epochs=5000, batch_size=501, shuffle=True, verbose=0, callbacks=callbackss)  
# history_dict = history.history
# with open(ResAnalyFile + '/history_'+str(9) +Strr+'.json', 'w') as f:
#         json.dump(history_dict, f)
# MinIndex=np.argmin(history.history['loss'])
# print('MinIndex:',MinIndex,'loss:',history.history['loss'][MinIndex],'R2:',history.history['R2'][MinIndex],'MAE:',history.history['MAE'][MinIndex],'MAPE:',history.history['MAPE'][MinIndex])

## Visualization    
# (1)real
Kernel_real=Cond_upkernel_1to40[13,:,:] 
Strr='_theoretical'
model = load_model(ResAnalyFile + '/best_model_'+str(9)+ Strr +'.h5', custom_objects={'R2': R2}) 
Midfiture_label = Model(inputs=model.input,outputs=model.get_layer('lambda').output)
Upkernel_1to40pred = Midfiture_label.predict([Data_low_1to40[0:14,:],np.linspace(1,40,14)])
Kernel_pred = Upkernel_1to40pred[13,:,:]
vmins =np.min(np.vstack((Kernel_real, Kernel_pred)))
vmaxs = np.max(np.vstack((Kernel_real, Kernel_pred)))
aa = np.min(Kernel_real-Kernel_pred)
bb = np.max(Kernel_real-Kernel_pred)
plt.figure(figsize=(15, 4))
plt.subplots_adjust(top=0.975,bottom=0.180,left=0.065,right=0.99,hspace=0.390,wspace=0.275)        
[X,Y]=np.meshgrid(np.linspace(-250,250,501),np.linspace(-250,250,501))
ax = plt.subplot(1, 3, 1) 
pltt = ax.contourf(X, Y, Kernel_pred,vmin=0,vmax=0.008, levels=np.linspace(0,0.008,100), cmap='RdBu_r', extend='both')
cbar = plt.colorbar(pltt, orientation='vertical',extend='both', pad=0.05,  location='right')
cbar.ax.set_position([cbar.ax.get_position().x0-0.008 ,cbar.ax.get_position().y0 , cbar.ax.get_position().width+1.5, cbar.ax.get_position().height * 0.915])
cbar.ax.tick_params(labelsize=18) 
cbar.ax.set_title('Weight', fontsize=18, ha='left', va='top')
ticks = np.linspace(0,0.008 ,5)
cbar.set_ticks(ticks)
cbar.ax.set_yticklabels([f"{tick:.3f}" for tick in ticks])
pltt2 = ax.contour(X, Y, Kernel_real,levels=50, cmap='RdBu_r',linewdiths=1)
ax.set_xticks(np.linspace(-250, 250, 6))
ax.set_yticks(np.linspace(-250, 250, 6))
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.set_xlabel('Points_upward',font={'family':'Times New Roman'},fontsize=20)
ax.set_ylabel('Points_obs',font={'family':'Times New Roman'},fontsize=20)
plt.text(0.01, 0.91, f"(a)", transform=plt.gca().transAxes, fontsize=25, font={'family':'Times New Roman'}, weight='bold')
# (2)Pred
ax2 = plt.subplot(1, 3, 2) 
[X,Y]=np.meshgrid(np.linspace(-250,250,501),np.linspace(-250,250,501))
pltt = ax2.contourf(X, Y, Kernel_pred,vmin=0,vmax=0.008, levels=np.linspace(0,0.008,100), cmap='RdBu_r', extend='both')
cbar = plt.colorbar(pltt, orientation='vertical',extend='both', pad=0.05,  location='right')
cbar.ax.set_position([cbar.ax.get_position().x0-0.008 ,cbar.ax.get_position().y0 , cbar.ax.get_position().width+1.5, cbar.ax.get_position().height * 0.915])
cbar.ax.tick_params(labelsize=18)  
cbar.ax.set_title('Weight', fontsize=18, ha='left', va='top')
ticks = np.linspace(0,0.008 ,5)
cbar.set_ticks(ticks)
cbar.ax.set_yticklabels([f"{tick:.3f}" for tick in ticks]) 
ax2.set_xticks(np.linspace(-250, 250, 6))
ax2.set_yticks(np.linspace(-250, 250, 6))
ax2.tick_params(axis='x', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)
ax2.set_xlabel('Points_upward',font={'family':'Times New Roman'},fontsize=20)
ax2.set_ylabel('Points_obs',font={'family':'Times New Roman'},fontsize=20)
plt.text(0.01, 0.915, f"(b)", transform=plt.gca().transAxes, fontsize=22, font={'family':'Times New Roman'}, weight='bold')
# (3)error
ax3 = plt.subplot(1, 3, 3) 
[X,Y]=np.meshgrid(np.linspace(-250,250,501),np.linspace(-250,250,501))
pltt = ax3.contourf(X, Y, Kernel_real- Kernel_pred, vmin=aa,vmax=bb,levels=100, cmap='RdBu_r', extend='both')
cbar = plt.colorbar(pltt, orientation='vertical', extend='both', pad=0.05,  location='right')
cbar.ax.set_position([cbar.ax.get_position().x0-0.008 ,cbar.ax.get_position().y0 , cbar.ax.get_position().width+1.5, cbar.ax.get_position().height * 0.915])
cbar.ax.tick_params(labelsize=18)  
cbar.ax.set_title('Weight', fontsize=18, ha='left', va='top')
ticks = np.linspace(aa,bb ,5)
cbar.set_ticks(ticks)
cbar.ax.set_yticklabels([f"{tick:.3f}" for tick in ticks])  
ax3.set_xticks(np.linspace(-250, 250, 6))
ax3.set_yticks(np.linspace(-250, 250, 6))
ax3.tick_params(axis='x', labelsize=20)
ax3.tick_params(axis='y', labelsize=20)
ax3.set_xlabel('Points_upward',font={'family':'Times New Roman'},fontsize=20)
ax3.set_ylabel('Points_obs',font={'family':'Times New Roman'},fontsize=20)
plt.text(0.01, 0.91, f"(c)", transform=plt.gca().transAxes, fontsize=25, font={'family':'Times New Roman'}, weight='bold')
plt.show()
save_path = ResAnalyFile.replace('Theoretical_weight/Allpoint/', '')
plt.savefig(save_path + '/Figure10_Theoretical_weight_all_results.png', dpi=300)





 