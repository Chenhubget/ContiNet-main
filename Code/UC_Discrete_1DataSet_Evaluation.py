from Utils.Packages import *
from Utils.Auxiliary import *
from Utils.ContiNet_model import *

# Load data
parent_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)).replace('\\', '/')
DataFile = parent_directory +'/Data_1UTrainSet4_Part1/'
ResAnalyFile = parent_directory + '/Results_1DiscreteUC_1DataSet_Evaluation/'    
Cond_upkernel = np.array(h5py.File(DataFile + 'Cond_upkernel_1to40.mat')['Cond_upkernel']).T
Weight_real = Cond_upkernel[:,:,250]

# Visualize and compare the results of different datasets
fig = plt.figure(figsize=(14,5.5))  
plt.subplots_adjust(top=0.905,bottom=0.140,left=0.090,right=0.990,hspace=0.200,wspace=0.200)
Colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
Markers = ['o','s','^','D','*','x']
for i in range(6):
    if i < 4:
        with open(ResAnalyFile + 'UC_Discrete_PredUCkernel_UTrainSet'+str(i+1) +'_Model_Layers9_CalPNum1.json','r') as f:
            Upkernel_pred = json.load(f)
    if i==4:
        with open(ResAnalyFile + 'UC_Discrete_PredUCkernel_UTrainSet'+str(i+1) +'_Noise1_Model_Layers9_CalPNum1.json','r') as f:
            Upkernel_pred = json.load(f)
    if i==5:
        with open(ResAnalyFile + 'UC_Discrete_PredUCkernel_UTrainSet'+str(i) +'_Noise2_Model_Layers9_CalPNum1.json','r') as f:
            Upkernel_pred = json.load(f)    
    Upkernel_pred = np.array(Upkernel_pred)
    mse=[];mae=[];r2=[]
    for j in range(0,Weight_real.shape[0]):
        mse.append(mean_squared_error(Weight_real[j,:],Upkernel_pred[j,:]))
        mae.append(mean_absolute_error(Weight_real[j,:],Upkernel_pred[j,:]))
        r2.append(r2_score(Weight_real[j,:],Upkernel_pred[j,:]))
    Data = [mae, r2]
    Title =['(a)','(b)']; Ylabel =['MAE(nT)','R²'] 
    Legend_dataset = ['UTrainSet1', 'UTrainSet2', 'UTrainSet3', 'UTrainSet4', 'UTrainSet5', 'UTrainSet6']
    for k in range(2):
        ax = plt.subplot(1,2,k+1)
        ax.plot(np.linspace(1,40,14,dtype='int'),Data[k],label=Legend_dataset[i],color=Colors[i],marker=Markers[i],markersize=3,linestyle='-',linewidth=1.5)
        if k==0:
            weizhi = 0.042
        else:
            weizhi = 0.042
        ax.text(weizhi, 0.95, Title[k], ha='center',va='center',transform=ax.transAxes,fontdict={'family':'Times New Roman','size':26,'weight':'bold'})
        ax.set_xlabel("k (Multiple of h)",font={'family':'Times New Roman'},fontsize=24)
        ax.set_ylabel(Ylabel[k],font={'family':'Times New Roman'},fontsize=24)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5)) 
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=7)) 
        if k==1:
            ax.set_yticks(np.linspace(0.5,1,6))
        ax.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%g')) 
        if k==0:
            ax.set_yticks(np.linspace(0,0.002,6))
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if k <=0:
            plt.legend(by_label.values(), by_label.keys(), fontsize=21, loc='upper right', bbox_to_anchor=(1.024, 1.025), handletextpad=0.1, labelspacing=0.2, frameon=False, ncol=2, columnspacing=1.2)
        else:
            plt.legend(by_label.values(), by_label.keys(), fontsize=21, loc='lower left', bbox_to_anchor=(-.024, -0.04), handletextpad=0.1, labelspacing=0.2,frameon=False)
        plt.grid(True, alpha=0.3)
        ax.set_xlim(1,40)
plt.show()
plt.savefig(ResAnalyFile + 'Figure6_Evaluation_TrainSets.png',dpi=300)
        
