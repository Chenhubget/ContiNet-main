from Utils.Packages import *
from Utils.Auxiliary import *
from Utils.ContiNet_model import *

# visualization and prediction
plt.figure(figsize=(15, 10))
# (1)Kolbeinseyridge
plt.subplot(2, 1, 1)
parent_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)).replace('\\', '/')
ResAnalyFile = parent_directory+'/Results_Actualcase/Kolbeinseyridge/'
Data = np.loadtxt(ResAnalyFile+'Kolbeinsey_profileData.dat', delimiter=',', skiprows=1) 
aaa = np.linspace(0,np.max(Data[:,0]),501)
Data_interp=np.zeros((501,3))
Data_interp[:,0] = aaa
for i in [1,2]:
    Data_interp[:,i] = np.interp(aaa,Data[:,0],Data[:,i])
np.savetxt(ResAnalyFile+'Kolbeinsey_profileData_interp.dat', Data_interp[:,[0,2]], delimiter=',')
Colors = ['#d62728', '#9467bd', '#e377c2','#17becf','#2ca02c']   #
DataFile =  parent_directory +'/Data_4DTrainSet_Part/'    
Mean_high = np.load(DataFile + 'Mean_high.npy'); Std_high  = np.load(DataFile + 'Std_high.npy')   
Mean_low = np.load(DataFile + 'Mean_low.npy'); Std_low  = np.load(DataFile + 'Std_low.npy')   
model = load_model(parent_directory +'/Results_3Theoretical_DC/Theoretical_1TrainingResults/DC_Theoretical_Modelparam_DTrainSet1and2_WithReg_Model_Layers9_CalPNumall' +'.h5',custom_objects={'R2':R2})
Data_low = Data_interp[:,1]; Data_high = Data_interp[:,2]
Data_high_normed = (Data_high - Mean_high) / Std_high
Cond_downheight = np.array([(4000-152.4)/471.34887111738])  
label_embedding =  (Cond_downheight /30).astype(np.float64)
Mae_kji = np.zeros((6,1)); R2_kji = np.zeros((6,1))
Data_lowpred2 = model.predict([Data_high_normed.reshape(1,501),label_embedding.reshape(1,1)])
Data_lowpred2 = Data_lowpred2 * Std_low + Mean_low
plt.plot(aaa/1000,Data_high,c='k',label= 'Emag2(V3)_4km', linewidth=1.5,zorder=4)
plt.plot(aaa/1000,Data_low,c='#1f77b4',label= 'AeroMagnetic', linewidth=1.5,zorder=3)
plt.plot(aaa/1000,Data_lowpred2.reshape(501,),c='#ff7f0e',label='DnContiNet', linewidth=1.5,zorder=2)
Data_lowpred2 = Data_lowpred2.reshape(501,)
Data_low = Data_low.reshape(501,)
Mae_kji[0] = np.mean(np.abs(Data_lowpred2 - Data_low));  R2_kji[0] = 1 - np.sum((Data_lowpred2 - Data_low)**2)/np.sum((Data_low - np.mean(Data_low))**2)
for i in range(5):
    Regcont_pred = np.loadtxt(ResAnalyFile+'REGCONT_Kolbeinsey/Kolbeinsey_profileData_interp_DCreg_3848_'+['C','L2','L1','L07','L05'][i]  +'.dat', delimiter=',',skiprows=0) 
    plt.plot(aaa/1000,Regcont_pred[:,1].reshape(501,),c=Colors[i],label='Regcont_'+['C','L2','L1','L07','L05'][i], linewidth=1.5,zorder=2)
    Mae_kji[i+1] = np.mean(np.abs(Regcont_pred[:,1].reshape(501,) - Data_low))
    R2_kji[i+1] = 1 - np.sum((Regcont_pred[:,1].reshape(501,) - Data_low)**2)/np.sum((Data_low - np.mean(Data_low))**2)
np.savetxt(ResAnalyFile+'Kolbeinsey_Mae_R2.txt', np.hstack((Mae_kji,R2_kji)), fmt='%.6f', header='MAE\tR2', delimiter='\t')
handles, labels = plt.gca().get_legend_handles_labels()
legend1 = plt.legend(handles[:3],labels[:3],loc='upper right', fontsize=15, frameon=False, bbox_to_anchor=(1+0.01, 1.03), borderaxespad=0.3, labelspacing=0.2, handlelength=1.5, handletextpad=0.3)
legend2 = plt.legend(handles[3:5], labels[3:5], loc='lower left', fontsize=15, frameon=False, bbox_to_anchor=(0.62, -0.03), borderaxespad=0.3, labelspacing=0.2, handlelength=1.5, handletextpad=0.3)
legend3 = plt.legend(handles[5:], labels[5:], loc='lower right', fontsize=15, frameon=False, bbox_to_anchor=(1.01, -0.03), borderaxespad=0.3, labelspacing=0.2, handlelength=1.5, handletextpad=0.3)
plt.gca().add_artist(legend2)
plt.gca().add_artist(legend1)
plt.xticks(font={'family':'Times New Roman'},fontsize=18)
plt.locator_params(axis='x', nbins=13)
plt.yticks(font={'family':'Times New Roman'},fontsize=18)
ax = plt.gca()
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
plt.text(0.008, 0.88, "(a)", transform=plt.gca().transAxes,fontsize=25, font={'family':'Times New Roman'}, weight='bold',zorder=4)
plt.xlabel('Distance(km)', font={'family': 'Times New Roman'}, fontsize=20)
plt.ylabel('Anomaly(nT)', font={'family': 'Times New Roman'}, fontsize=20)
plt.xlim(0,240)

# (2)Mohnsridge
plt.subplot(2, 1, 2)
ResAnalyFile = parent_directory +'/Results_Actualcase/Mohnsridge/'
Data = np.loadtxt(ResAnalyFile+'Mohns_Profiledata.dat', delimiter=',', skiprows=1) 
aaa = np.linspace(0,np.max(Data[:,0]),501)
Data_interp=np.zeros((501,3))
Data_interp[:,0] = aaa
for i in [1,2]:
    Data_interp[:,i] = np.interp(aaa,Data[:,0],Data[:,i])
np.savetxt(ResAnalyFile+'Mohns_Profiledata_interp.dat', Data_interp[:,[0,2]], delimiter=',')
Data_low = Data_interp[:,1]; Data_high = Data_interp[:,2]
Data_high_normed = (Data_high - Mean_high) / Std_high
Cond_downheight = np.array([(4000-300)/902.130060149])  
label_embedding =  (Cond_downheight /30).astype(np.float64)
Mae_mji = np.zeros((6,1)); R2_Mji = np.zeros((6,1))
Data_lowpred2 = model.predict([Data_high_normed.reshape(1,501),label_embedding.reshape(1,1)])
Data_lowpred2 = Data_lowpred2 * Std_low + Mean_low
plt.plot(aaa/1000,Data_high,c='k',label= 'Emag2(V3)_4km', linewidth=1.5,zorder=4)
plt.plot(aaa/1000,Data_low,c='#1f77b4',label= 'AeroMagnetic', linewidth=1.5,zorder=3)
plt.plot(aaa/1000,Data_lowpred2.reshape(501,),c='#ff7f0e',label='DnContiNet', linewidth=1.5,zorder=2)
Data_lowpred2 = Data_lowpred2.reshape(501,)
Data_low = Data_low.reshape(501,)
Mae_mji[0]=  np.mean(np.abs(Data_lowpred2 - Data_low));  R2_Mji[0] = 1 - np.sum((Data_lowpred2 - Data_low)**2)/np.sum((Data_low - np.mean(Data_low))**2)
for i in range(5):
    Regcont_pred = np.loadtxt(ResAnalyFile+'REGCONT_Mohns/Mohns_Profiledata_interp_DCreg_3700_'+['C','L2','L1','L07','L05'][i]  +'.dat', delimiter=',',skiprows=0) 
    plt.plot(aaa/1000,Regcont_pred[:,1].reshape(501,),c=Colors[i],label='Regcont_'+['C','L2','L1','L07','L05'][i], linewidth=1.5,zorder=2)
    Mae_mji[i+1] = np.mean(np.abs(Regcont_pred[:,1].reshape(501,) - Data_low))
    R2_Mji[i+1] = 1 - np.sum((Regcont_pred[:,1].reshape(501,) - Data_low)**2)/np.sum((Data_low - np.mean(Data_low))**2)
np.savetxt(ResAnalyFile+'Mohns_Mae_R2.txt', np.hstack((Mae_mji,R2_Mji)), fmt='%.6f', header='MAE\tR2', delimiter='\t')
handles, labels = plt.gca().get_legend_handles_labels()
legend1 = plt.legend(handles[:2], labels[:2], loc='upper right', fontsize=15, frameon=False, bbox_to_anchor=(1+0.01, 1.03), borderaxespad=0.3, labelspacing=0.2, handlelength=1.5, handletextpad=0.3)
legend2 = plt.legend(handles[2:5], labels[2:5], loc='lower left', fontsize=15, frameon=False, bbox_to_anchor=(0.62, -0.03), borderaxespad=0.3, labelspacing=0.2, handlelength=1.5, handletextpad=0.3)
legend3 = plt.legend(handles[5:], labels[5:], loc='lower right', fontsize=15, frameon=False, bbox_to_anchor=(1.01, -0.03), borderaxespad=0.3, labelspacing=0.2, handlelength=1.5, handletextpad=0.3)
plt.gca().add_artist(legend2)
plt.gca().add_artist(legend1)
plt.xticks(font={'family':'Times New Roman'},fontsize=18)
plt.locator_params(axis='x', nbins=10)
plt.yticks(font={'family':'Times New Roman'},fontsize=18)
ax = plt.gca()
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
plt.text(0.008, 0.87, "(b)", transform=plt.gca().transAxes,fontsize=25, font={'family':'Times New Roman'}, weight='bold',zorder=4)
plt.xlabel('Distance(km)', font={'family': 'Times New Roman'}, fontsize=20)
plt.ylabel('Anomaly(nT)', font={'family': 'Times New Roman'}, fontsize=20)
plt.xlim(0,460)
plt.subplots_adjust(top=0.985,bottom=0.110,left=0.105,right=0.980,hspace=0.290,wspace=0.205)
plt.show()
save_path = ResAnalyFile.replace('Mohnsridge/', '')
plt.savefig(save_path + 'Figure20_Comparison_profile_results.png', dpi=300)
 