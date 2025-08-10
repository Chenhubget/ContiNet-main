from Utils.Packages import *
from Utils.Auxiliary import *
from Utils.ContiNet_model import *

# Load data
parent_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)).replace('\\', '/')
DataFile =  parent_directory +'/Data_4DTrainSet_Part/'    
ResAnalyFile= parent_directory +'/Results_3Theoretical_DC/'
# Data_low1 = np.array(h5py.File(DataFile + 'Data_ZLow_single.mat')['Data_ZLow']).T
# Data_high1 = np.array(h5py.File(DataFile + 'Data_ZHigh_single.mat')['Data_ZHigh']).T
# Cond_downheight1 = np.array(h5py.File(DataFile + 'Cond_Downheight_single.mat')['Cond_Downheight']).T
# Data_low2 = np.array(h5py.File(DataFile + 'Data_ZLow_fuhe.mat')['Data_ZLow']).T
# Data_high2 = np.array(h5py.File(DataFile + 'Data_ZHigh_fuhe.mat')['Data_ZHigh']).T
# Cond_downheight2 = np.array(h5py.File(DataFile + 'Cond_Downheight_fuhe.mat')['Cond_Downheight']).T   
# Data_low = np.concatenate((Data_low1,Data_low2),axis=0)
# Data_high = np.concatenate((Data_high1,Data_high2),axis=0)
# Cond_downheight = np.concatenate((Cond_downheight1,Cond_downheight2),axis=0)

# Normalization 
# Mean_high = np.mean(Data_high,axis=0); Std_high =  np.std(Data_high,axis=0)
# Mean_low = np.mean(Data_low,axis=0); Std_low = np.std(Data_low,axis=0)
# np.save(DataFile + 'Mean_high.npy', Mean_high); np.save(DataFile + 'Std_high.npy', Std_high)
# np.save(DataFile + 'Mean_low.npy', Mean_low); np.save(DataFile + 'Std_low.npy', Std_low)
# Data_high_normed = (Data_high - np.mean(Data_high,axis=0)) / np.std(Data_high,axis=0)
# Data_low_normed = (Data_low - np.mean(Data_low,axis=0)) / np.std(Data_low,axis=0)
# label_embedding = Cond_downheight / np.max(Cond_downheight,axis=0) 

## Visualization 
# (1)Loss, R2 
File1 = '/Theoretical_1TrainingResults/DC_Theoretical_Modelparam_DTrainSet1and2_NoReg_Model_Layers9_CalPNumall-epoch_'
File2 = '/Theoretical_1TrainingResults/DC_Theoretical_Modelparam_DTrainSet1and2_WithReg_Model_Layers9_CalPNumall-epoch_'
Trainindex = ['loss','R2']; Index2 = ['Loss_UpContiNet', 'R²_UpContiNet']; Index3 = ['Loss_DnContiNet', 'R²_DnContiNet']
Colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.figure(figsize=(12,4) )   
for i in range(2):
    Data2 = pd.read_csv(ResAnalyFile+File1+Trainindex[i]+'.csv')
    Data3 = pd.read_csv(ResAnalyFile+File2+Trainindex[i]+'.csv')
    ax1 = plt.subplot(1, 2, i+1)
    l1, = ax1.plot(Data2['Step'], Data2['Value'], '--',marker='o', label=Index2[i], color=Colors[0],linewidth=1.5,markersize=3)
    ax1.set_xlabel('Epoch', fontsize=18)
    ax1.set_ylabel(Index2[i], fontsize=18)
    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    ax1.set_xticks(np.linspace(0, 500, 6))
    ax2 = ax1.twinx()
    l2, = ax2.plot(Data3['Step'], Data3['Value'],'-', marker='*', label=Index3[i], color=Colors[1],linewidth=1.5,markersize=3)
    ax2.set_ylabel(Index3[i], rotation=270,fontsize=18,labelpad=20)
    ax2.tick_params(axis='y', labelsize=18)
    ax2.set_xticks(np.linspace(0, 500, 6))
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    if i==0:
        ax1.legend(lines, labels, loc='upper right', fontsize=16, frameon=False, bbox_to_anchor=(1.02,1.02))
    if i==1:
        ax1.legend(lines, labels, loc='lower right', fontsize=16, frameon=False, bbox_to_anchor=(1.02,-0.02))
    plt.text(0.01, 0.912, f"({chr(96 + i+1)})", transform=ax1.transAxes, fontsize=25, font={'family':'Times New Roman'},weight='bold')
plt.subplots_adjust(top=0.975,bottom=0.165,left=0.085,right=0.925,hspace=0.345,wspace=0.485)   
plt.show()
plt.savefig(ResAnalyFile+'/Theoretical_1TrainingResults/Figure12_Comparison_different_DnContiNet_structures.png', dpi=300, bbox_inches='tight')
plt.close()

# (2)Weight fitting results of different methods
Upkernel = np.array(h5py.File(DataFile + 'Cond_upkernel.mat')['Cond_upkernel']).T
plt.figure(figsize=(16, 12))
plt.subplots_adjust(top=0.975,bottom=0.075,left=0.065,right=0.965,hspace=0.390,wspace=0.275)
Dnhigh = [5,15,30]
for i in range(3):
    Upkernel_inv = np.linalg.inv(Upkernel[Dnhigh[i]-1,:,:])  
    Upkernel_inv = Upkernel_inv[:,int(501/2):251]  # 取中间251个点  
    # Index = np.where(Cond_downheight[:,0]==Dnhigh[i])[0]
    # Train_x =  Data_high_normed[Index,:]; Train_y = Data_low_normed[Index,:]; Train_label = label_embedding[Index,:]
    # model = load_model(ResAnalyFile + '/Theoretical_1TrainingResults/DC_Theoretical_Modelparam_DTrainSet1and2_NoReg_Model_Layers9_CalPNumall.h5',custom_objects={'R2':R2})
    # Midfiture_label2 = Model(inputs=model.input,outputs=model.get_layer('lambda').output)
    # Kernel_pred2 = Midfiture_label2.predict([Train_x[:1,:],Train_label[:1,:]]).reshape(501,501) 
    # model = load_model(ResAnalyFile + '/Theoretical_1TrainingResults/DC_Theoretical_Modelparam_DTrainSet1and2_WithReg_Model_Layers9_CalPNumall.h5',custom_objects={'R2':R2})
    # Midfiture_label3 = Model(inputs=model.input,outputs=model.get_layer('lambda').output)
    # Kernel_pred3 = Midfiture_label3.predict([Train_x[:1,:],Train_label[:1,:]]).reshape(501,501)
    # np.save(ResAnalyFile + f'/Theoretical_2Weight_fitting_results/Kernel_NoReg_Dnhigh{Dnhigh[i]}.npy', Kernel_pred2)
    # np.save(ResAnalyFile + f'/Theoretical_2Weight_fitting_results/Kernel_WithReg_Dnhigh{Dnhigh[i]}.npy', Kernel_pred3)
    # 加载保存的权重
    Kernel_pred2 = np.load(ResAnalyFile + f'/Theoretical_2Weight_fitting_results/Kernel_NoReg_Dnhigh{Dnhigh[i]}.npy')
    Kernel_pred3single = np.load(ResAnalyFile + f'/Theoretical_2Weight_fitting_results/Kernel_WithReg_Dnhigh{Dnhigh[i]}.npy')
    Data_temp = np.concatenate((Upkernel_inv,Kernel_pred2[:,int(501/2):251],Kernel_pred3single[:,int(501/2):251]),axis=1).T
    Label = ['Kernel_inv','UpContiNet','DnContiNet(Single)','DnContiNet(All)']
    for j in range(3):
       ax = plt.subplot(3,3,j*3+i+1)
       if j!=3:
            ax.plot(np.linspace(-250, 250, 501),Data_temp[j],c='#1f77b4',label='Weight_'+str(Dnhigh[i]))
            ax.set_xlabel('Points_obs',font={'family':'Times New Roman'},fontsize=15)
            ax.set_ylabel('Weight',font={'family':'Times New Roman'},fontsize=15)
            ax.set_xticks(np.linspace(-250, 250, 6))
            ax.set_yticks(ax.get_yticks())
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            ax = plt.gca() 
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))   
            ax.yaxis.get_offset_text().set_fontsize(15)  
            ax.yaxis.set_offset_position('left')   
            ax.set_xlim(-250, 250)
       plt.text(0.016, 0.87, f"({chr(96+ j*3+i+1)})", transform=plt.gca().transAxes, fontsize=18, font={'family':'Times New Roman'}, weight='bold')
       plt.legend(loc='upper right',fontsize=13,frameon=False ,bbox_to_anchor=(1.03,1.05), handletextpad=0.2,bbox_transform=plt.gca().transAxes)   
       if j*3+i+1 in [3,6,9,12,15]:
            ax2 = ax.twinx()   
            if j==3:
                paddd = 55
            if j!=3:
                paddd = 18
            ax2.set_ylabel(Label[j], fontdict={'family': 'Times New Roman'}, fontsize=15,rotation=270,labelpad=paddd)
            ax2.yaxis.set_label_position("right")   
            ax2.yaxis.tick_right()  
            ax2.set_yticks([])   
plt.show()
plt.savefig(ResAnalyFile + '/Theoretical_2Weight_fitting_results/Figure13_Weight_different_methods.png', dpi=300)
plt.close()

# (3)DnContiNet weight matrix
Upkernel = np.array(h5py.File(DataFile + 'Cond_upkernel.mat')['Cond_upkernel']).T
plt.figure(figsize=(15, 4))
plt.subplots_adjust(top=0.975,bottom=0.180,left=0.065,right=0.99,hspace=0.390,wspace=0.275)
Dnhigh = [5,15,30]
for i in range(3):
    Kernel_pred3 = np.load(ResAnalyFile + f'/Theoretical_2Weight_fitting_results/Kernel_WithReg_Dnhigh{Dnhigh[i]}.npy')
    ax = plt.subplot(1,3,i+1)
    [X,Y]=np.meshgrid(np.linspace(-250,250,501),np.linspace(-250,250,501))
    vmin = np.min(Kernel_pred3); vmax = np.max(Kernel_pred3)  
    pltt = ax.contourf(X, Y, Kernel_pred3, levels=np.linspace(vmin,vmax,100), cmap='RdBu_r', extend='both')
    cbar = plt.colorbar(pltt, orientation='vertical', extend='both', pad=0.05,  location='right')
    cbar.ax.set_position([cbar.ax.get_position().x0-0.008 ,cbar.ax.get_position().y0 , cbar.ax.get_position().width+1.5, cbar.ax.get_position().height * 0.915])
    cbar.ax.tick_params(labelsize=18) 
    cbar.ax.set_title('Weight', fontsize=18, ha='left', va='top')
    ticks = np.linspace(np.min(Kernel_pred3), np.max(Kernel_pred3), 5)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{tick:.2f}" for tick in ticks])  
    ax.set_xticks(np.linspace(-250, 250, 6))
    ax.set_yticks(np.linspace(-250, 250, 6))
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xlabel('Points_downward',font={'family':'Times New Roman'},fontsize=20)
    ax.set_ylabel('Points_obs',font={'family':'Times New Roman'},fontsize=20)
    plt.text(0.01, 0.915, f"({chr(96+i+1)})", transform=plt.gca().transAxes, fontsize=25, font={'family':'Times New Roman'}, weight='bold')
plt.show()
plt.savefig(ResAnalyFile + '/Theoretical_2Weight_fitting_results/Figure14_Kernelmatrix_DnContiNet.png', dpi=300)
plt.close()

# (4)Training samples 
Colors = ['k','#d62728', '#ff7f0e','#1f77b4', '#9467bd']     
fig = plt.figure(figsize=(14,5))
Dnhigh = [5,15,25]
for i in range(3):
    plt.subplot(1,2,1)
    plt.text(0.01, 0.93, '(a)', transform=plt.gca().transAxes,fontsize=25, font={'family':'Times New Roman'}, weight='bold')
    # Index = np.where(Cond_downheight1[:,0]==Dnhigh[i])[0][9001]   
    if i == 0:
        # Data_high1sample = Data_high1[Index,:]
        # np.save(ResAnalyFile + f'/Theoretical_3TrainingSamples/Data_high1sample_Dnhigh{Dnhigh[i]}.npy', Data_high1sample)
        Data_high1sample = np.load(ResAnalyFile + f'/Theoretical_3TrainingSamples/Data_high1sample_Dnhigh{Dnhigh[i]}.npy')
        plt.plot(np.linspace(-250, 250, 501),Data_high1sample,c=Colors[i],label= 'RealFwd_30', linewidth=3,zorder=2)
    # Data_low1sample =  Data_low1[Index,:]
    # np.save(ResAnalyFile + f'/Theoretical_3TrainingSamples/Data_low1sample_Dnhigh{Dnhigh[i]}.npy', Data_low1sample)
    Data_low1sample = np.load(ResAnalyFile + f'/Theoretical_3TrainingSamples/Data_low1sample_Dnhigh{Dnhigh[i]}.npy')
    plt.plot(np.linspace(-250, 250, 501),Data_low1sample,c=Colors[i+1],label= 'RealFwd_'+ str(int(30-Dnhigh[i])), linewidth=3,zorder=2,linestyle='--')
    plt.xlabel('Points_obs',font={'family':'Times New Roman'},fontsize=25)
    plt.ylabel('Anomaly(nT)',font={'family':'Times New Roman'},fontsize=25)
    plt.xticks(np.linspace(-250, 250, 6),font={'family':'Times New Roman'},fontsize=22)
    plt.yticks(font={'family':'Times New Roman'},fontsize=22)
    ax = plt.gca()
    ax.set_xlim(-250, 250)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
    plt.legend(loc='upper right',fontsize=17,frameon=False,bbox_to_anchor=(1.02,1.02))
    plt.subplot(1,2,2)
    plt.text(0.01, 0.93, '(b)', transform=plt.gca().transAxes,fontsize=25, font={'family':'Times New Roman'}, weight='bold')
    # Index = np.where(Cond_downheight2[:,0]==Dnhigh[i])[0][431]  
    if i == 0: 
        # Data_high2sample = Data_high2[Index,:]
        Data_high2sample = np.load(ResAnalyFile + f'/Theoretical_3TrainingSamples/Data_high2sample_Dnhigh{Dnhigh[i]}.npy')
        # np.save(ResAnalyFile + f'/Theoretical_3TrainingSamples/Data_high2sample_Dnhigh{Dnhigh[i]}.npy', Data_high2sample)
        plt.plot(np.linspace(-250, 250, 501),Data_high2sample,c=Colors[i],label= 'RealFwd_30', linewidth=3,zorder=2)
    # Data_low2sample = Data_low2[Index,:]
    # np.save(ResAnalyFile + f'/Theoretical_3TrainingSamples/Data_low2sample_Dnhigh{Dnhigh[i]}.npy', Data_low2sample)
    Data_low2sample = np.load(ResAnalyFile + f'/Theoretical_3TrainingSamples/Data_low2sample_Dnhigh{Dnhigh[i]}.npy')
    plt.plot(np.linspace(-250, 250, 501),Data_low2sample,c=Colors[i+1],label= 'RealFwd_'+ str(int(30-Dnhigh[i])), linewidth=3,zorder=2,linestyle='--')
    
    plt.xlabel('Points_obs',font={'family':'Times New Roman'},fontsize=25)
    plt.ylabel('Anomaly(nT)',font={'family':'Times New Roman'},fontsize=25)
    plt.xticks(np.linspace(-250, 250, 6),font={'family':'Times New Roman'},fontsize=22)
    plt.yticks(font={'family':'Times New Roman'},fontsize=22)
    ax = plt.gca()
    ax.set_xlim(-250, 250)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
    plt.legend(loc='upper right', fontsize=17, frameon=False, bbox_to_anchor=(1.02, 1.02))
plt.subplots_adjust(top=0.97,bottom=0.16,left=0.095,right=0.99,hspace=0.17,wspace=0.255)
plt.show()
plt.savefig(ResAnalyFile + '/Theoretical_3TrainingSamples/Figure15_Training_samples.png', dpi=300)
plt.close()

# (5)Training results
Colors = ['k','#d62728', '#ff7f0e','#2ca02c', '#17becf','#1f77b4', '#9467bd']  
model_N = load_model(ResAnalyFile + '/Theoretical_1TrainingResults/DC_Theoretical_Modelparam_DTrainSet1and2_NoReg_Model_Layers9_CalPNumall.h5',custom_objects={'R2':R2})
model_W = load_model(ResAnalyFile + '/Theoretical_1TrainingResults/DC_Theoretical_Modelparam_DTrainSet1and2_WithReg_Model_Layers9_CalPNumall.h5',custom_objects={'R2':R2})
Midfiture_out_N = Model(inputs=model_N.input,outputs=model_N.get_layer('tf.reshape_1').output)
Midfiture_out_W = Model(inputs=model_W.input,outputs=model_W.get_layer('tf.reshape_1').output)
Midfiture_outadd = Model(inputs=model_W.input,outputs=model_W.get_layer('dense_14').output)
Dnhigh = [5,5,15,15,25,25]
Sbplt=[1,2,3,4,5,6]
titleindex = [1,4,2,5,3,6]
fig = plt.figure(figsize=(19, 20))
for i in range(6):  
    # Indexx = np.where(Cond_downheight[:,0]==Dnhigh[i])[0]
    # Index = np.array([Indexx[9001],Indexx[18001]])    
    # Out_N = Midfiture_out_N.predict([Data_high_normed[Index,:],label_embedding[Index,:]])
    # Out_W = Midfiture_out_W.predict([Data_high_normed[Index,:],label_embedding[Index,:]])
    # Out_Add = Midfiture_outadd.predict([Data_high_normed[Index,:],label_embedding[Index,:]])
    # Data_lowpred = model_W.predict([Data_high_normed[Index,:],label_embedding[Index,:]])    
    # Out_N = Out_N * np.std(Data_low,axis=0) + np.mean(Data_low,axis=0)
    # Out_W = Out_W * np.std(Data_low,axis=0) + np.mean(Data_low,axis=0)
    # Out_Add = Out_Add * np.std(Data_low,axis=0) + np.mean(Data_low,axis=0)
    # Data_lowpred = Data_lowpred * np.std(Data_low,axis=0) + np.mean(Data_low,axis=0)    
    ax = plt.subplot(3,2,Sbplt[i])
    ax.set_xlim(-250,250)
    ax.text(0.011, 0.89, f"({chr(96 + titleindex[i])})", transform=plt.gca().transAxes,fontsize=25, font={'family':'Times New Roman'}, weight='bold')
    if Sbplt[i] in [1,3,5]: 
        Dim = 0
    else: 
        Dim = 1
    # # 保存数据
    # np.save(ResAnalyFile + f'/Theoretical_4TrainingSamplesAnalysis/Out_N_Sbplt{Sbplt[i]}.npy', Out_N[Dim,:])
    # np.save(ResAnalyFile + f'/Theoretical_4TrainingSamplesAnalysis/Out_W_Sbplt{Sbplt[i]}.npy', Out_W[Dim,:])
    # np.save(ResAnalyFile + f'/Theoretical_4TrainingSamplesAnalysis/Out_Add_Sbplt{Sbplt[i]}.npy', Out_Add[Dim,:])
    # np.save(ResAnalyFile + f'/Theoretical_4TrainingSamplesAnalysis/Data_lowpred_Sbplt{Sbplt[i]}.npy', Data_lowpred[Dim,:])
    # np.save(ResAnalyFile + f'/Theoretical_4TrainingSamplesAnalysis/Data_low_Sbplt{Sbplt[i]}.npy', Data_low[Index,:][Dim,:])
    # np.save(ResAnalyFile + f'/Theoretical_4TrainingSamplesAnalysis/label_embedding_Sbplt{Sbplt[i]}.npy', label_embedding[Index,:][Dim,:])
    # 加载数据
    Out_N_temp = np.load(ResAnalyFile + f'/Theoretical_4TrainingSamplesAnalysis/Out_N_Sbplt{Sbplt[i]}.npy')
    Out_W_temp = np.load(ResAnalyFile + f'/Theoretical_4TrainingSamplesAnalysis/Out_W_Sbplt{Sbplt[i]}.npy')
    Out_Add_temp = np.load(ResAnalyFile + f'/Theoretical_4TrainingSamplesAnalysis/Out_Add_Sbplt{Sbplt[i]}.npy')     
    Data_lowpred_temp = np.load(ResAnalyFile + f'/Theoretical_4TrainingSamplesAnalysis/Data_lowpred_Sbplt{Sbplt[i]}.npy')
    Data_low_temp = np.load(ResAnalyFile + f'/Theoretical_4TrainingSamplesAnalysis/Data_low_Sbplt{Sbplt[i]}.npy')
    label_embedding_temp = np.load(ResAnalyFile + f'/Theoretical_4TrainingSamplesAnalysis/label_embedding_Sbplt{Sbplt[i]}.npy')
    ax.plot(np.linspace(-250, 250, 501),Data_low_temp,c='#1f77b4',linestyle='-',label= 'RealFwd_' +str(int(30-Dnhigh[i])), linewidth=1.5,zorder=2)
    ax.plot(np.linspace(-250, 250, 501),Out_N_temp,c='#ff7f0e',linestyle='-',marker='o',markersize=0,label='UpContiNet_'+ str(int(30-Dnhigh[i])), linewidth=1.5,zorder=3)
    ax.plot(np.linspace(-250, 250, 501),Data_lowpred_temp,c='#9467bd',linestyle='-',marker='*',markersize=0,label='DnContiNet_'+ str(int(30-Dnhigh[i])), linewidth=1.5,zorder=4)
    ax.plot(np.linspace(-250, 250, 501),Out_W_temp,c='#2ca02c',linestyle='-',marker='^',markersize=0,label='DnMainNet_'+ str(int(30-Dnhigh[i])), linewidth=1.5,zorder=3)
    ax.plot(np.linspace(-250, 250, 501),label_embedding_temp * Out_Add_temp,linestyle='-.', c='#d62728',label='DnRegNet_'+ str(int(30-Dnhigh[i])),linewidth=1.5,zorder=1)
    if titleindex[i] in [4,5]:
        handles, labels = ax.get_legend_handles_labels()
        legend1 = ax.legend(handles[:5], labels[:5], loc='upper right', fontsize=14, frameon=False, bbox_to_anchor=(1.01, 1.03),labelspacing=0.4,handlelength=2,handletextpad=0.5)
        legend2 = ax.legend(handles[5:6], labels[5:6], loc='lower right', fontsize=14, frameon=False, bbox_to_anchor=(1.01, -0.04),labelspacing=0.4,handlelength=2,handletextpad=0.5)
        ax.add_artist(legend1)
    if titleindex[i] in [3]:
        handles, labels = ax.get_legend_handles_labels()
        legend1 = ax.legend(handles[:5], labels[:5], loc='upper right', fontsize=14, frameon=False, bbox_to_anchor=(1.01, 1.03),labelspacing=0.4,handlelength=2,handletextpad=0.5)
        legend2 = ax.legend(handles[5:6], labels[5:6], loc='lower right', fontsize=14, frameon=False, bbox_to_anchor=(1.01, -0.04),labelspacing=0.4,handlelength=2,handletextpad=0.5)
        ax.add_artist(legend1)
    if titleindex[i] in [1,2,6]:
        ax.legend(loc='upper right', fontsize=14, frameon=False, bbox_to_anchor=(1.01, 1.03),labelspacing=0.4,handlelength=2,handletextpad=0.5)
    if i==2:
       ax.set_ylabel('Anomaly(nT)',font={'family':'Times New Roman'},fontsize=23)
    plt.xticks(np.linspace(-250, 250, 6),font={'family':'Times New Roman'},fontsize=20)
    plt.yticks(font={'family':'Times New Roman'},fontsize=20)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
fig.text(0.55, 0.015, 'Points_obs', font={'family': 'Times New Roman'}, fontsize=23, ha='center', va='center')
plt.subplots_adjust(top=0.985,bottom=0.07,left=0.105,right=0.980,hspace=0.17,wspace=0.195)
plt.show()
plt.savefig(ResAnalyFile + '/Theoretical_4TrainingSamplesAnalysis/Figure16_Comparison_DCtraining_results.png', dpi=300)






 