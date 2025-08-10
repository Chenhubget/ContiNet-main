from Utils.Packages import *
from Utils.Auxiliary import *
from Utils.ContiNet_model import *

# Load data
parent_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)).replace('\\', '/')
ResAnalyFile = parent_directory + '/Results_2TheoreticalUC_2AllPoints/'
DataFile = parent_directory+ '/Data_3UTestSet/'      
DataSet_ZLow1 = np.array(h5py.File(DataFile + 'Data_ZLow1.mat')['Data_ZLow']).T
DataSet_ZHigh1 = np.array(h5py.File(DataFile + 'Data_ZHigh1.mat')['Data_ZHigh']).T
DataSet_ZUp1 = np.array(h5py.File(DataFile + 'Data_ZUp1.mat')['Data_ZUp']).T
DataSet_ZLow2 = np.array(h5py.File(DataFile + 'Data_ZLow.mat')['Data_ZLow']).T
DataSet_ZHigh2 = np.array(h5py.File(DataFile + 'Data_ZHigh.mat')['Data_ZHigh']).T
DataSet_ZUp2 = np.array(h5py.File(DataFile + 'Data_ZUp.mat')['Data_ZUp']).T
Cond_upheight = np.linspace(1, 40, 40).reshape(-1, 1)

# Prediction and visualization
model = load_model(ResAnalyFile + 'UC_Theoretical_Modelparam_UTrainSet4_Concatenate_AllPoints_Model_Layers9_CalPNum1.h5', custom_objects={'R2': R2}) 
predictions1 = model.predict([DataSet_ZLow1, Cond_upheight]); predictions2 = model.predict([DataSet_ZLow2, Cond_upheight])
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(22, 12))
# (1)Single anomaly prediction
for  i in [0,5,10,15,20,25,30,35,39]: 
     ax1.plot(np.linspace(-250, 250, 501), DataSet_ZHigh1[i, :], c='b', linestyle='-', linewidth=1.5, label='RealFwd', markersize=4)
     ax1.plot(np.linspace(-250, 250, 501), DataSet_ZUp1[i, :], c='g', linestyle='-', linewidth=1.5, label='RealUpwd', markersize=4)
     ax1.plot(np.linspace(-250, 250, 501), predictions1[i, :], c='#ff7f0e', linestyle='--',dashes=(5,5), linewidth=1.5, label='PredFwd', markersize=4)
     ax1.plot(np.linspace(-250, 250, 501), DataSet_ZHigh1[i, :] - predictions1[i, :], c='k', linestyle='-.', linewidth=1.5, label='FwdError', markersize=4)
ax1.text(0.01, 0.98, '(a)', transform=ax1.transAxes, fontsize=25, verticalalignment='top', horizontalalignment='left',weight='bold')
ax1.set_ylabel('Anomaly(nT)', fontsize=22)
ax1.set_xlabel('Points_obs', fontsize=22)
ax1.tick_params(axis='y', labelsize=22)
ax1.tick_params(axis='x', labelsize=22)
ax1.legend(['RealFwd','RealUpwd','PredFwd','FwdError'],fontsize=18, loc='upper right', bbox_to_anchor=(1.01, 1.02), handletextpad=0.1, labelspacing=0.3, frameon=False)
ax1.yaxis.set_major_formatter(ScalarFormatter())
ax1.yaxis.set_major_locator(plt.MaxNLocator(6))
ax1.set_xlim(-250, 250)
ax1.set_ylim(-50, 80)
ax1.set_xticks([-250, -150, -50, 0, 50, 150, 250])
# (2)Single anomaly observation
ax3.plot(np.linspace(-250, 250, 501), DataSet_ZLow1[0, :], c='k', linestyle='-', linewidth=1.5, label='ObsFwd',marker='o', markersize=4)
ax3.text(0.01, 0.98, '(b)', transform=ax3.transAxes, fontsize=25, verticalalignment='top', horizontalalignment='left',weight='bold')
ax3.set_ylabel('Anomaly(nT)', fontsize=24)
ax3.set_xlabel('Points_obs', fontsize=24)
ax3.tick_params(axis='y', labelsize=22)
ax3.tick_params(axis='x', labelsize=22)
ax3.set_ylim(-60, 100)
ax3.yaxis.set_major_formatter(ScalarFormatter())
ax3.set_xlim(-250, 250)
ax3.set_xticks([-250, -150, -50, 0, 50, 150, 250])
ax3.yaxis.set_major_locator(plt.MaxNLocator(6))
ax3.legend(fontsize=18, loc='upper right',  handletextpad=0.1, labelspacing=0.3,  bbox_to_anchor=(1.01, 1.02),frameon=False)
# (3)Compex anomaly prediction
for  i in [0,5,10,15,20,25,30,35,39]:  
     ax2.plot(np.linspace(-250, 250, 501), DataSet_ZHigh2[i, :], c='b', linestyle='-', linewidth=1.5, label='RealFwd')
     ax2.plot(np.linspace(-250, 250, 501), DataSet_ZUp2[i, :], c='g', linestyle='-', linewidth=1.5, label='RealUpwd')
     ax2.plot(np.linspace(-250, 250, 501), predictions2[i, :], c='#ff7f0e', linestyle='--',dashes=(5,5), linewidth=1.5, label='PredFwd')
     ax2.plot(np.linspace(-250, 250, 501), DataSet_ZHigh2[i, :] - predictions2[i, :], c='k', linestyle='-.', linewidth=1.5, label='FwdError')
ax2.text(0.01, 0.98, '(c)', transform=ax2.transAxes, fontsize=25, verticalalignment='top', horizontalalignment='left',weight='bold')
ax2.set_ylabel('Anomaly(nT)', fontsize=24)
ax2.set_xlabel('Points_obs', fontsize=24)
ax2.tick_params(axis='y', labelsize=22)
ax2.tick_params(axis='x', labelsize=22)
ax2.yaxis.set_major_formatter(ScalarFormatter())
ax2.yaxis.set_major_locator(plt.MaxNLocator(6))
ax2.set_ylim(-180, 220)
ax2.set_xlim(-250, 250)
ax2.set_xticks([-250, -150, -50, 0, 50, 150, 250])
handles, labels = ax2.get_legend_handles_labels()
legend1 = ax2.legend(handles[:3], labels[:3], loc='upper right', fontsize=18, frameon=False, bbox_to_anchor=(1.01, 1.02), handletextpad=0.1, labelspacing=0.3)
legend2 = ax2.legend(handles[3:4], labels[3:4], loc='lower right', fontsize=18, frameon=False, bbox_to_anchor=(1.01, -0.04), handletextpad=0.1, labelspacing=0.3)
ax2.add_artist(legend1)
# (4)Comlex anomaly observation
ax4.plot(np.linspace(-250, 250, 501), DataSet_ZLow2[0, :], c='k', linestyle='-', linewidth=1.5, label='ObsFwd',marker='o', markersize=4)
ax4.text(0.01, 0.98, '(d)', transform=ax4.transAxes, fontsize=25, verticalalignment='top', horizontalalignment='left',weight='bold')
ax4.set_ylabel('Anomaly(nT)', fontsize=24)
ax4.set_xlabel('Points_obs', fontsize=24)
ax4.tick_params(axis='y', labelsize=22)
ax4.tick_params(axis='x', labelsize=22)
ax4.set_ylim(-180, 220)
ax4.set_xlim(-250, 250)
ax4.set_xticks([-250, -150, -50, 0, 50, 150, 250])
ax4.yaxis.set_major_formatter(ScalarFormatter())
ax4.yaxis.set_major_locator(plt.MaxNLocator(6))
ax4.legend(fontsize=18, loc='upper right', handletextpad=0.1, labelspacing=0.3,  bbox_to_anchor=(1.01, 1.02),frameon=False)
plt.subplots_adjust(top=0.975,bottom=0.095,left=0.070,right=0.980,hspace=0.295,wspace=0.190)
plt.show()
plt.savefig(ResAnalyFile + '/Figure11_Theoretical_simulation.png', dpi=300, bbox_inches='tight')






 