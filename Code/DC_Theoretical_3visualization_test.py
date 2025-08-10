from Utils.Packages import *
from Utils.Auxiliary import *
from Utils.ContiNet_model import *

# Load Data
parent_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)).replace('\\', '/')
DataFile1 = parent_directory + '/Data_4DTrainSet_Part/'
DataFile2 = parent_directory + '/Data_5DTestSet/' 
ResAnalyFile= parent_directory + '/Results_3Theoretical_DC/'
Colors = ['k','#d62728', '#ff7f0e','#2ca02c', '#17becf','#1f77b4', '#9467bd',]    
Mean_high = np.load(DataFile1 + 'Mean_high.npy'); Std_high  = np.load(DataFile1 + 'Std_high.npy')   
Mean_low = np.load(DataFile1 + 'Mean_low.npy'); Std_low  = np.load(DataFile1 + 'Std_low.npy')    
model = load_model(ResAnalyFile + '/Theoretical_1TrainingResults/DC_Theoretical_Modelparam_DTrainSet1and2_WithReg_Model_Layers9_CalPNumall' +'.h5',custom_objects={'R2':R2})

# (1)Visualization test samples 
Regcont = pd.read_csv(DataFile2 + 'fig2_output_all_calculated_data.dat')   
Regcont=np.array(Regcont).T
x_original = np.linspace(0, Regcont.shape[1] - 1, Regcont.shape[1])
x_new = np.linspace(0, Regcont.shape[1] - 1, 501)
Regcont_interpolated = np.zeros((Regcont.shape[0], 501))
for i in range(Regcont.shape[0]):
    interpolator = interp1d(x_original, Regcont[i, :], kind='linear')
    Regcont_interpolated[i, :] = interpolator(x_new)
Regcont = Regcont_interpolated
Data_high1 = Regcont[1]
Cond_downheight1 = np.array([[1000], [2000], [3000], [4000], [4500], [4800]])  /200
Data_high1_normed = (Data_high1 - Mean_high) / Std_high
label_embedding1 =  (Cond_downheight1 /30).astype(np.float64)
Data_high2 = np.array(h5py.File(DataFile2 + 'Data_ZHigh_fuhe_test.mat')['Data_ZHigh']).T
Data_low2 = np.array(h5py.File(DataFile2 + 'Data_ZLow_fuhe_test.mat')['Data_ZLow']).T
Cond_downheight2 = np.array(h5py.File(DataFile2 + 'Cond_Downheight_fuhe_test.mat')['Cond_Downheight']).T   
indexx = np.array([0,1,2,3,4,5])
Data_high2 =Data_high2[indexx,:]; Data_low2 = Data_low2[indexx,:]; Cond_downheight2 = Cond_downheight2[indexx,:]
Data_high2_normed =  (Data_high2 - Mean_high) / Std_high
Data_low2_normed = (Data_low2 - Mean_low) / Std_low
label_embedding2 = Cond_downheight2 / 30
fig = plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.text(0.01, 0.93, '(a)', transform=plt.gca().transAxes,fontsize=28, font={'family':'Times New Roman'},weight='bold')
plt.plot(Regcont[0,:],Data_high1,c=Colors[0],label= 'Real_0m', linewidth=3.5,zorder=2)
for i in range(6):
    plt.plot(Regcont[0,:],Regcont[2+i,:],c=Colors[i+1],label= 'Real_-'+ str(int(Cond_downheight1[i]*200))+'m', linewidth=3,zorder=2,linestyle='--')
plt.xlabel('Points_obs',font={'family':'Times New Roman'},fontsize=25)
plt.ylabel('Anomaly(nT)',font={'family':'Times New Roman'},fontsize=25)
plt.xticks(font={'family':'Times New Roman'},fontsize=22)
plt.yticks(font={'family':'Times New Roman'},fontsize=22)
plt.xlim(0, 100000)
ax = plt.gca()
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
plt.legend(loc='upper right',fontsize=17,frameon=False,bbox_to_anchor=(1, 1+0.02))
plt.subplot(1,2,2)
plt.text(0.01, 0.93, '(b)', transform=plt.gca().transAxes,fontsize=28, font={'family':'Times New Roman'},weight='bold')
plt.plot(np.linspace(-250, 250, 501),Data_high2[0,:],c=Colors[0],label= 'RealFwd_30', linewidth=4,zorder=2)
for i in range(6):
    plt.plot(np.linspace(-250, 250, 501),Data_low2[i,:],c=Colors[i+1],label= 'RealFwd_'+ str(30-Cond_downheight2[i][0]), linewidth=3,zorder=2,linestyle='--')
plt.xlabel('Points_obs',font={'family':'Times New Roman'},fontsize=25)
plt.ylabel('Anomaly(nT)',font={'family':'Times New Roman'},fontsize=25)
plt.xticks(np.linspace(-250, 250, 6),font={'family':'Times New Roman'},fontsize=22)
plt.yticks(font={'family':'Times New Roman'},fontsize=22)
ax = plt.gca()
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
handles, labels = plt.gca().get_legend_handles_labels()
legend1 = plt.legend(handles[:3],labels[:3],loc='upper right', fontsize=17, frameon=False, bbox_to_anchor=(1, 1+0.02))
legend2 = plt.legend(handles[3:],labels[3:], loc='lower left', fontsize=17, frameon=False, bbox_to_anchor=(0,-0.03))
plt.gca().add_artist(legend1)
plt.xlim(-250, 250)
plt.subplots_adjust(top=0.985,bottom=0.14,left=0.07,right=0.98,hspace=0.195,wspace=0.17)
plt.show()
# plt.savefig(ResAnalyFile + '/Theoretical_5TestSamples/Figure17_Testsamples.png', dpi=300)
plt.close()

# (2)Prediction test samples 
Regcont = pd.read_csv(DataFile2 + 'fig3_output_all_calculated_data.dat')   
Regcont=np.array(Regcont).T
x_original = np.linspace(0, Regcont.shape[1] - 1, Regcont.shape[1])
x_new = np.linspace(0, Regcont.shape[1] - 1, 501)
Regcont_interpolated = np.zeros((Regcont.shape[0], 501))
for i in range(Regcont.shape[0]):
    interpolator = interp1d(x_original, Regcont[i, :], kind='linear')
    Regcont_interpolated[i, :] = interpolator(x_new)
Regcont = Regcont_interpolated
Data_high1 = Regcont[1]
Cond_downheight1 = np.array([[1000], [3000], [4500], [4800]])  /200
Data_high1_normed = (Data_high1 - Mean_high) / Std_high
label_embedding1 =  (Cond_downheight1 /30).astype(np.float64)
Data_high2 = np.array(h5py.File(DataFile2 + 'Data_ZHigh_fuhe_test.mat')['Data_ZHigh']).T
Data_low2 = np.array(h5py.File(DataFile2 + 'Data_ZLow_fuhe_test.mat')['Data_ZLow']).T
Cond_downheight2 = np.array(h5py.File(DataFile2 + 'Cond_Downheight_fuhe_test.mat')['Cond_Downheight']).T   
indexx = np.array([0,2,4,5])
Data_high2 =Data_high2[indexx,:]; Data_low2 = Data_low2[indexx,:]; Cond_downheight2 = Cond_downheight2[indexx,:]
Data_high2_normed =  (Data_high2 - Mean_high) / Std_high
Data_low2_normed = (Data_low2 - Mean_low) / Std_low
label_embedding2 = Cond_downheight2 / 30
Testpred = pd.read_excel(ResAnalyFile + '/REGCONT_DC\L05_norm/Testpred_Regcot1.xlsx'); Testpred =np.array(Testpred).T
Testpred=Testpred[indexx,:]
fig = plt.figure(figsize=(12, 12))
plt.subplots_adjust(top=0.985,bottom=0.065,left=0.085,right=0.98,hspace=0.220,wspace=0.255)
Subpltindex = [1,3,5,7]
for i in range(4):
    plt.subplot(4,2,Subpltindex[i])
    Data_high1_normed = Data_high1_normed.reshape(1,501); condition =label_embedding1[i].reshape(1,1)
    Data_lowpred1 = model.predict([Data_high1_normed,condition])
    Data_lowpred1 = Data_lowpred1 * Std_low + Mean_low
    plt.plot(Regcont[0,:],Regcont[2+i,:],c='#1f77b4',linestyle='-',label= 'Real_-'+str(int(Cond_downheight1[i]*200))+'m', linewidth=1.5,zorder=2)
    plt.plot(Regcont[0,:],Regcont[6+i,:],c='#ff7f0e',linestyle='-',marker='o',markersize=0,label=  'REGCONT_-'+str(int(Cond_downheight1[i]*200))+'m', linewidth=1.5,zorder=2)
    plt.plot(Regcont[0,:],Data_lowpred1.reshape(501,),c='#9467bd',linestyle='-',marker='*',markersize=0,label='DnContiNet_-'+str(int(Cond_downheight1[i]*200))+'m', linewidth=1.5,zorder=4)
    plt.legend(loc='upper right',fontsize=9,frameon=False, bbox_to_anchor=(1+0.02, 1+0.02))
    plt.xticks(font={'family':'Times New Roman'},fontsize=13)
    plt.yticks(font={'family':'Times New Roman'},fontsize=13)
    ax = plt.gca()
    ax.set_xlim(0, 103000)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
    plt.text(0.01, 0.867, f"({chr(96 + i+1)})", transform=plt.gca().transAxes,fontsize=16, font={'family':'Times New Roman'}, weight='bold',zorder=9)
Subpltindex = [2,4,6,8]
for i in range(4):
    plt.subplot(4,2,Subpltindex[i])
    Data_lowpred2 = model.predict([Data_high2_normed[i,:].reshape(1,501),label_embedding2[i,:].reshape(1,1)])
    Data_lowpred2 = Data_lowpred2 * Std_low + Mean_low
    plt.plot(np.linspace(-250, 250, 501),Data_low2[i,:],c='#1f77b4',linestyle='-',label= 'RealFwd_'+str(30-Cond_downheight2[i][0]), linewidth=1.5,zorder=2)
    plt.plot(np.linspace(-250, 250, 501), Testpred[i, :], c='#ff7f0e', linestyle='-',marker='o', markersize=0, label='REGCONT_'+str(30-Cond_downheight2[i][0]), linewidth=1.5, zorder=2)
    plt.plot(np.linspace(-250, 250, 501),Data_lowpred2.reshape(501,),c='#9467bd',linestyle='-',marker='*',markersize=0,label='DnContiNet_'+str(30-Cond_downheight2[i][0]), linewidth=1.5,zorder=4)
    plt.xticks(np.linspace(-250, 250, 6),font={'family':'Times New Roman'},fontsize=13)
    plt.yticks(font={'family':'Times New Roman'},fontsize=13)
    plt.xlim(-250, 250)
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
    plt.text(0.01, 0.867, f"({chr(96 + 4+i+1)})", transform=plt.gca().transAxes,fontsize=16, font={'family':'Times New Roman'}, weight='bold',zorder=9)
    handles, labels = plt.gca().get_legend_handles_labels()
    legend1 = plt.legend(handles[:3],labels[:3],loc='upper right', fontsize=9, frameon=False, bbox_to_anchor=(1+0.02, 1+0.02))
    legend2 = plt.legend(handles[3:],labels[3:], loc='lower right', fontsize=9, frameon=False, bbox_to_anchor=(1.02,-0.03))
    plt.gca().add_artist(legend1)
fig.text(0.015, 0.52, 'Anomaly(mGal)', font={'family': 'Times New Roman'}, fontsize=16, rotation='vertical', ha='center', va='center')
fig.text(0.52, 0.52, 'Anomaly(nT)', font={'family': 'Times New Roman'}, fontsize=16, rotation='vertical', ha='center', va='center')
plt.show()
plt.savefig(ResAnalyFile + '/Theoretical_5TestSamples/Figure18_SampleData_testpredict.png', dpi=300)







 