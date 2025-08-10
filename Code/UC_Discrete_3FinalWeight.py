from Utils.Packages import *
from Utils.Auxiliary import *
from Utils.ContiNet_model import *

## Load data
parent_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)).replace('\\', '/')
ResAnalyFile = parent_directory + '/Results_1DiscreteUC_3FinalWeight/'
Cond_upkernel_1to40 = np.array(h5py.File(parent_directory + '/Data_1UTrainSet4_Part1/'  + 'Cond_upkernel_1to40.mat')['Cond_upkernel']).T
Cond_upkernel_2to38 = np.array(h5py.File(parent_directory + '/Data_1UTrainSet4_Part1/'  + 'Cond_upkernel_2to38.mat')['Cond_upkernel']).T
Cond_upkernel_3to39 = np.array(h5py.File(parent_directory + '/Data_1UTrainSet4_Part1/'  + 'Cond_upkernel_3to39.mat')['Cond_upkernel']).T
Cond_upkernel_1p5to39p5 = np.array(h5py.File(parent_directory + '/Data_2UTrainSet4_Part2/'  + 'Cond_upkernel_1p5to39p5.mat')['Cond_upkernel']).T
Cond_upkernel = np.vstack((Cond_upkernel_1to40,Cond_upkernel_2to38,Cond_upkernel_3to39,Cond_upkernel_1p5to39p5))

## Visualization
model = load_model(ResAnalyFile + 'UC_Discrete_Modelparam_UTrainSet4_Concatenate_Model_Layers9_CalPNum1.h5',custom_objects={'R2':R2}) 
Midfiture_label = Model(inputs=model.input,outputs=model.get_layer('dense_8').output)
Upkernel_1to40pred = Midfiture_label.predict([np.ones([14,501,1]),np.linspace(1,40,14)])
Upkernel_2to38pred = Midfiture_label.predict([np.ones([13,501,1]),np.linspace(2,38,13)])
Upkernel_3to39pred = Midfiture_label.predict([np.ones([13,501,1]),np.linspace(3,39,13)])
Upkernel_1p5to39p5pred = Midfiture_label.predict([np.ones([39,501,1]),np.linspace(1.5,39.5,39)])
Kernel_pred1 = np.vstack((Upkernel_1to40pred,Upkernel_2to38pred,Upkernel_3to39pred))
Kernel_pred2 = Upkernel_1p5to39p5pred
kernel_pred = np.vstack((Kernel_pred1,Kernel_pred2))
# (1)weight
plt.figure(figsize=(22,9))
plt.subplots_adjust(top=0.930,bottom=0.135,left=0.090,right=0.975,hspace=0.214,wspace=0.205)
ax = plt.subplot(1,2,1)
Colors = ['b','#2ca02c', '#ff7f0e']
for j in range(np.shape(Cond_upkernel)[0]):
    ax.plot(np.linspace(-250,250,501),Cond_upkernel[j,:,250],c='b',linestyle='-',linewidth=1)
for j in range(np.shape(Kernel_pred1)[0]):
    ax.plot(np.linspace(-250,250,501),Kernel_pred1[j,:],c='#ff7f0e',linestyle='--',dashes=(6, 6),linewidth=1.5)
for j in range(np.shape(Kernel_pred2)[0]):
    ax.plot(np.linspace(-250,250,501),Kernel_pred2[j,:],c='#2ca02c',linestyle='--',dashes=(6, 6),linewidth=1.5)
for j in range(np.shape(Cond_upkernel)[0]):
    ax.plot(np.linspace(-250,250,501),Cond_upkernel[j,:,250]-kernel_pred[j,:],c='k',linestyle='-.',linewidth=1.5)
ax.plot([], [], c='b', linestyle='-', linewidth=1, label='Weight_discrete')
ax.plot([], [], c='#ff7f0e', linestyle='--', linewidth=1.5, label='Weight_trainpred')
ax.plot([], [], c='#2ca02c', linestyle='--', linewidth=1.5, label='Weight_testpred')
ax.plot([], [], c='k', linestyle='-.', linewidth=1.5, label='Weight_error')
ax.text(0.045,0.95,'(a)',transform=ax.transAxes,fontsize=25,fontweight='bold',horizontalalignment='center',verticalalignment='center')
plt.ylabel('Weight',fontsize=25)
plt.xlabel('Points_obs',fontsize=25)
plt.xticks([-250,-150,-50,0,50,150,250],fontsize=22)
plt.xlim(-250,250)
Minn = np.min([np.min(Cond_upkernel[:,:,250]),np.min(kernel_pred),np.min(Cond_upkernel[:,:,250] - kernel_pred)])
Maxx = np.max([np.max(Cond_upkernel[:,:,250]),np.max(kernel_pred),np.max(Cond_upkernel[:,:,250] - kernel_pred)])
plt.yticks(np.linspace(Minn,Maxx,7),fontsize=22)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.legend(fontsize=20,loc='upper right',bbox_to_anchor=(1.02, 1.017),handletextpad=0.1,labelspacing=0.3,frameon=False)
zoom_ax = plt.axes([ax.get_position().x0 +0.003, ax.get_position().y0 +0.12, 0.192, 0.6])  
for j in range(np.shape(Cond_upkernel)[0]):
    zoom_ax.plot(np.linspace(-30, 30, 61), Cond_upkernel[j,220:281,250], ls='-', color='b', linewidth=1)
for j in range(np.shape(Kernel_pred1)[0]):
    zoom_ax.plot(np.linspace(-30, 30, 61), Kernel_pred1[j,220:281], ls='--', color='#ff7f0e',dashes=(6, 6), linewidth=1.5)  
for j in range(np.shape(Kernel_pred2)[0]):
    zoom_ax.plot(np.linspace(-30, 30, 61), Kernel_pred2[j,220:281], ls='--', color='#2ca02c', dashes=(6, 6),linewidth=1.5)
for j in range(np.shape(Cond_upkernel)[0]):
    zoom_ax.plot(np.linspace(-30, 30, 61), Cond_upkernel[j,220:281,250] - kernel_pred[j,220:281], ls='-.', color='k', linewidth=1.5)
zoom_ax.set_xlim(-30, 30)
zoom_ax.set_ylim(-0.01, 0.08)
zoom_ax.tick_params(axis='both', which='both', labelsize=0, length=0)  
zoom_ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
zoom_ax.patch.set_alpha(0)  
zoom_ax.axis('off')  
arrow_x = -35 
arrow_y = 0.02 
zoom_center_x = -110  
zoom_center_y = 0.034  
ax.annotate('', xy=(arrow_x, arrow_y), xytext=(zoom_center_x, zoom_center_y), arrowprops=dict(facecolor='#8B4513', edgecolor='#8B4513', arrowstyle='->', lw=2, mutation_scale=20))

# (2)SDP
ax = plt.subplot(1,2,2)
x_true = Cond_upkernel[7,:,250]; x_pred = kernel_pred[7,:]
R22=pearsonr(x_true, x_pred).statistic
MAEE = mean_absolute_error(x_true, x_pred)
MSEE = mean_squared_error(x_true, x_pred)  
def f_1(x,A,B):
    return A*x + B
slope,intercept = optimize.curve_fit(f_1,x_true,x_pred)[0]
y_fit = slope*x_true + intercept
n = 1 
t_value = 1.96 
std_err = np.std(x_pred - y_fit)
margin_of_error = t_value * (std_err / np.sqrt(n))
lower_confidence_bound = slope * x_true + intercept - margin_of_error
upper_confidence_bound = slope * x_true + intercept + margin_of_error
ax.plot(x_true, lower_confidence_bound,linewidth=1.8, linestyle='--', color='k', dashes=(1, 2), label='95% Prediction Band')
ax.plot(x_true, upper_confidence_bound, linewidth=1.8,linestyle='--', color='k', dashes=(1, 2))
fitlabel='FitLine:'+str("{:.1f}".format(slope))+'*x'+'+'+str("{:.1f}".format(intercept))
if intercept<0:
        fitlabel='FitLine:'+str("{:.1f}".format(slope))+'*x'+str("{:.1f}".format(intercept))
ax.plot(x_true,y_fit,color='k',linewidth=1.8,linestyle='-',label=fitlabel)
ax.plot(x_true,x_true,color='r',linewidth=1.8,linestyle='--',label="1:1 Line")
xy = np.vstack([x_true,x_pred])    
z = gaussian_kde(xy)(xy)            
idx = z.argsort()                
x, y, z = x_true[idx], x_pred[idx], z[idx] 
z_nomalized=(z-np.min(z))/(np.max(z)-np.min(z))
scatter=ax.scatter(x,y,marker='o',c=z_nomalized,s=15,label='Data Point',cmap='Spectral_r',vmin=0,vmax=1)  
cbar=plt.colorbar(scatter,orientation='vertical',extend='both',pad=0.015,aspect=40,shrink=1)
cbar.ax.set_position([cbar.ax.get_position().x0+0.003,cbar.ax.get_position().y0 , cbar.ax.get_position().width+1.5, cbar.ax.get_position().height * 0.957])
cbar.ax.tick_params(labelsize=17,direction='in')
cbar.ax.set_title('Freq',fontsize=15,ha='left', va='top')
font_properties = {'family': 'Times New Roman', 'color':  'k', 'weight': 'normal', 'size': 19}
ax.text(0.97,0.03, f"{'R²=%.2f' % R22}", horizontalalignment='right', transform=ax.transAxes,fontdict=font_properties)
ax.text(0.97,0.11, f"{'MSE=%.2e' % MSEE}", horizontalalignment='right', transform=ax.transAxes,fontdict=font_properties)
ax.text(0.97,0.19, f"{'MAE=%.2e' % MAEE}",horizontalalignment='right', transform=ax.transAxes,fontdict=font_properties)
ax.legend(loc='upper left',fontsize=19,frameon = False,bbox_to_anchor=(-0.03, 0.96),handletextpad=0.3, labelspacing=0.43)
ax.text(0.055,0.95,'(b)',transform=ax.transAxes,fontsize=25,fontweight='bold',horizontalalignment='center',verticalalignment='center')
ax.set_xlabel('Weight_discrete',fontsize=25)
ax.set_ylabel('Weight_pred',fontsize=25)
fontdict2= {"color":"k",'family':'Times New Roman'}
ax.grid(True, linestyle='--', alpha=0.5)
ax.tick_params(axis='both', labelsize=20)
ax.set_xticks(np.linspace(0,0.015,6))
ax.set_yticks(np.linspace(0,0.015,6))
plt.show()
plt.savefig(ResAnalyFile + '/Figure8_Discrete_weight_results.png',dpi=300)




 