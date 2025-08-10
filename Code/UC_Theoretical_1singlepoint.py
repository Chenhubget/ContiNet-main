from Utils.Packages import *
from Utils.Auxiliary import *
from Utils.ContiNet_model import *

## Load data
parent_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)).replace('\\', '/')
ResAnalyFile= parent_directory +'/Results_2TheoreticalUC_1SinglePoint/'
Cond_upkernel_1to40 = np.array(h5py.File(parent_directory + '/Data_1UTrainSet4_Part1/'  + 'Cond_upkernel_1to40.mat')['Cond_upkernel']).T
Cond_upkernel_2to38 = np.array(h5py.File(parent_directory + '/Data_1UTrainSet4_Part1/'  + 'Cond_upkernel_2to38.mat')['Cond_upkernel']).T
Cond_upkernel_3to39 = np.array(h5py.File(parent_directory + '/Data_1UTrainSet4_Part1/'  + 'Cond_upkernel_3to39.mat')['Cond_upkernel']).T
Cond_upkernel = np.vstack((Cond_upkernel_1to40,Cond_upkernel_2to38,Cond_upkernel_3to39))

## Visualization
plt.figure(figsize=(13, 8))
plt.subplots_adjust(top=0.940,bottom=0.090,left=0.085,right=0.980,hspace=0.295,wspace=0.235)                                                                              
Upward_kernel=Cond_upkernel[:,:,250] 
with open(ResAnalyFile + 'weight_OLS.json', 'r') as f:
    weight_lstsq = np.array(json.load(f))
# (1)Lstsq   
plt.subplot(2,2,1)
plt.text(0.01,0.9,'(a)',transform=plt.gca().transAxes,fontsize=22,weight='bold')
plt.yticks(fontsize=20) 
plt.xlim(-250,250)
plt.xticks([-250,-150,-50,0,50,150,250],fontsize=30) 
k=13
plt.plot(np.linspace(-250,250,501),Upward_kernel[k,:],ls='-',color='b',linewidth=1.5,label='Weight_scatter')
plt.plot(np.linspace(-250,250,501),weight_lstsq[k,:],ls='--',color='#ff7f0e',dashes=(5, 5),linewidth=1.5,label='Weight_lstsq')
plt.plot(np.linspace(-250,250,501),Upward_kernel[k,:]-weight_lstsq[k,:],ls='-.',color='k',linewidth=1.5,label='Weight_error')
plt.plot([-50,-50],[np.min(Upward_kernel[k,:]),np.max(Upward_kernel[k,:])/2],linestyle='--',color='#2ca02c',linewidth=1.5,zorder=3)
plt.plot([50,50],[np.min(Upward_kernel[k,:]),np.max(Upward_kernel[k,:])/2],linestyle='--',color='#2ca02c',linewidth=1.5,zorder=3)
plt.legend(['Weight_discrete','Weight_OLS','Weight_error'],fontsize=17,loc='upper right',bbox_to_anchor=(0.997, 0.997),borderaxespad=0,handletextpad=0.1,labelspacing=0.3,frameon=False)
plt.ylabel('Weight',fontsize=20)
plt.xlabel('Points_obs',fontsize=20)
plt.xticks([-250,-150,-50,0,50,150,250],fontsize=20) 
plt.yticks(fontsize=20) 
plt.xlim(-250,250)
ax=plt.gca()
ax.yaxis.set_major_formatter(FormatStrFormatter('%1.3f'))
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5)) 
plt.text(-50,np.max(Upward_kernel[k,:])/2,'FLD',fontsize=15,horizontalalignment='center')
plt.text(50,np.max(Upward_kernel[k,:])/2,'LLD',fontsize=15,horizontalalignment='center')
ax = plt.gca()
ax.annotate('', xy=(-50, np.max(Upward_kernel[k,:])/3), xytext=(50, np.max(Upward_kernel[k,:])/3),arrowprops=dict(arrowstyle='<->', color='k', lw=1,ls='--'), va='center', ha='center')
plt.text(0, np.max(Upward_kernel[k,:])/3, 'LDiff', fontsize=15, horizontalalignment='center',verticalalignment='bottom')
ax.annotate('', xy=(-150, np.max(Upward_kernel[k,:])/4), xytext=(-50, np.max(Upward_kernel[k,:])/4),arrowprops=dict(arrowstyle='->', color='r', lw=1,ls='--'), va='center', ha='center')
ax.annotate('', xy=(50, np.max(Upward_kernel[k,:])/4), xytext=(150, np.max(Upward_kernel[k,:])/4),arrowprops=dict(arrowstyle='<-', color='r', lw=1,ls='--'), va='center', ha='center')
plt.text(-100, np.max(Upward_kernel[k,:])/4 , 'HDiff', fontsize=15, horizontalalignment='center',verticalalignment='bottom')
plt.text(100, np.max(Upward_kernel[k,:])/4, 'HDiff', fontsize=15, horizontalalignment='center',verticalalignment='bottom')
# (2)Lstsq all   
plt.subplot(2,2,2)
plt.text(0.01,0.9,'(b)',transform=plt.gca().transAxes,fontsize=22,weight='bold')
for k in range(Upward_kernel.shape[0]):
        plt.plot(np.linspace(-250,250,501),Upward_kernel[k,:],ls='-',color='b',linewidth=1.5,label='Weight_scatter')
        plt.plot(np.linspace(-250,250,501),weight_lstsq[k,:],ls='--',color='#ff7f0e',dashes=(5, 5),linewidth=1.5,label='Weight_lstsq')
        plt.plot(np.linspace(-250,250,501),Upward_kernel[k,:]-weight_lstsq[k,:],ls='-.',color='k',linewidth=1.5,label='Weight_error')
plt.legend(['Weight_discrete','Weight_OLS','Weight_error'],fontsize=17,loc='upper right',bbox_to_anchor=(0.997, 0.997),borderaxespad=0,handletextpad=0.1,labelspacing=0.3,frameon=False)
plt.ylabel('Weight',fontsize=20)
plt.xlabel('Points_obs',fontsize=20)
plt.xticks([-250,-150,-50,0,50,150,250],fontsize=20) 
plt.yticks([0,0.1,0.2,0.3],fontsize=20) 
plt.xlim(-250,250)
ax=plt.gca()
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax = plt.gca()
zoom_ax = plt.axes([plt.gca().get_position().x0 +0.003, plt.gca().get_position().y0 + 0.036, 0.191, 0.28])   
for k in range(Upward_kernel.shape[0]):
    zoom_ax.plot(np.linspace(-250,250,501),Upward_kernel[k,:],ls='-',color='b',linewidth=1.5,label='Weight_scatter')
    zoom_ax.plot(np.linspace(-250,250,501),weight_lstsq[k,:],ls='--',color='#ff7f0e',dashes=(5, 5),linewidth=1.5,label='Weight_lstsq')
    zoom_ax.plot(np.linspace(-250,250,501),Upward_kernel[k,:]-weight_lstsq[k,:],ls='-.',color='k',linewidth=1.5,label='Weight_error')
zoom_ax.set_xlim(-50, 50)
zoom_ax.set_ylim(-0.01, 0.05)
zoom_ax.set_xticks([-50, -25, 0, 25, 50])
zoom_ax.set_yticks([-0.0005, -0.00025, 0, 0.00025, 0.0005])
zoom_ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
zoom_ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
zoom_ax.patch.set_alpha(0)
zoom_ax.axis('off')  
arrow_x = -30 
arrow_y =0.025
zoom_center_x = -110 
zoom_center_y =0.04
ax.annotate('', xy=(arrow_x, arrow_y), xytext=(zoom_center_x, zoom_center_y), arrowprops=dict(facecolor='#8B4513', edgecolor='#8B4513', arrowstyle='->', lw=2, mutation_scale=20))
# (3)Upcontinet   
Strr='_theoretical'
model = load_model(ResAnalyFile + 'UC_Theoretical_Modelparam_UTrainSet4_Concatenate_SinglePoints_Model_Layers9_CalPNum1.h5',custom_objects={'R2':R2})
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=1e-6),metrics=[R2,MAE,MSE,MAPE])    
model.summary()
Colors = ['b','#2ca02c', '#ff7f0e']
Midfiture_label = Model(inputs=model.input,outputs=model.get_layer('dense_8').output)
Upkernel_1to40pred = Midfiture_label.predict([np.ones([14,501,1]),np.linspace(1,40,14)])
Upkernel_2to38pred = Midfiture_label.predict([np.ones([13,501,1]),np.linspace(2,38,13)])
Upkernel_3to39pred = Midfiture_label.predict([np.ones([13,501,1]),np.linspace(3,39,13)])
Kernel_pred = np.vstack((Upkernel_1to40pred,Upkernel_2to38pred,Upkernel_3to39pred))
plt.subplot(2,2,3)
j = 13
plt.plot(np.linspace(-250, 250, 501), Cond_upkernel[j, :, 250], c='b', linestyle='-', linewidth=1.5)
plt.plot(np.linspace(-250, 250, 501), Kernel_pred[j, :], c='#ff7f0e', dashes=(5,5), linestyle='--', linewidth=1.5)
plt.plot(np.linspace(-250, 250, 501), Cond_upkernel[j, :, 250] - Kernel_pred[j, :], c='k', linestyle='-.', linewidth=1.5)
plt.text(0.01,0.9,'(c)',transform=plt.gca().transAxes,fontsize=22,weight='bold')
plt.ylabel('Weight', fontsize=20)
plt.xlabel('Points_obs', fontsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xticks([-250,-150,-50,0,50,150,250],fontsize=20) 
Range= Cond_upkernel[j, :, 250]
plt.plot([-50,-50],[np.min(Range),np.max(Range)/2],'g--',lw=1)
plt.plot([50,50],[np.min(Range),np.max(Range)/2],'g--',lw=1)
plt.text(-50,np.max(Range)/2,'FLD',fontsize=15,horizontalalignment='center')
plt.text(50,np.max(Range)/2,'LLD',fontsize=15,horizontalalignment='center')
ax = plt.gca()
ax.annotate('', xy=(-50, np.max(Range)/3), xytext=(50, np.max(Range)/3), arrowprops=dict(arrowstyle='<->', color='k', lw=1, ls='--'), va='center', ha='center')
plt.text(0, np.max(Range)/3 , 'LDiff', fontsize=15, horizontalalignment='center',verticalalignment='bottom')
ax.annotate('', xy=(-150, np.max(Range)/4), xytext=(-50, np.max(Range)/4),arrowprops=dict(arrowstyle='->', color='r', lw=1,ls='--'), va='center', ha='center')
ax.annotate('', xy=(50, np.max(Range)/4), xytext=(150, np.max(Range)/4),arrowprops=dict(arrowstyle='<-', color='r', lw=1,ls='--'), va='center', ha='center')
plt.text(-100, np.max(Range)/4 , 'HDiff', fontsize=15, horizontalalignment='center',verticalalignment='bottom')
plt.text(100, np.max(Range)/4, 'HDiff', fontsize=15, horizontalalignment='center',verticalalignment='bottom')
plt.legend(['Weight_discrete', 'Weight_pred', 'Weight_error'], fontsize=17, loc='upper right', bbox_to_anchor=(1.008, 1.017), handletextpad=0.1, labelspacing=0.3,frameon=False)
plt.xticks([-250, -150, -50, 0, 50, 150, 250], fontsize=20)
plt.xlim(-250, 250)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# (4)Upcontinet all
plt.subplot(2,2,4)
for j in range(np.shape(Cond_upkernel)[0]):
    plt.plot(np.linspace(-250, 250, 501), Cond_upkernel[j, :, 250], c='b', linestyle='-', linewidth=1.5)
    plt.plot(np.linspace(-250, 250, 501), Kernel_pred[j, :], c='#ff7f0e',dashes=(5,5), linestyle='--', linewidth=1.5)
    plt.plot(np.linspace(-250, 250, 501), Cond_upkernel[j, :, 250] - Kernel_pred[j, :], c='k', linestyle='-.', linewidth=1.5)
plt.text(0.01,0.9,'(d)',transform=plt.gca().transAxes,fontsize=22,weight='bold')
plt.ylabel('Weight', fontsize=20)
plt.xlabel('Points_obs', fontsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.tick_params(axis='x', labelsize=20)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.xticks([-250,-150,-50,0,50,150,250],fontsize=20) 
plt.legend(['Weight_discrete', 'Weight_pred', 'Weight_error'], fontsize=17, loc='upper right', bbox_to_anchor=(1.008, 1.017), handletextpad=0.1, labelspacing=0.3,frameon=False)
plt.xlim(-250, 250)
ax = plt.gca()
zoom_ax = plt.axes([plt.gca().get_position().x0+0.003 , plt.gca().get_position().y0 +  0.036, 0.191, 0.28])    
for j in range(Cond_upkernel.shape[0]):
    zoom_ax.plot(np.linspace(-250, 250, 501), Cond_upkernel[j, :, 250], c='b', linestyle='-', linewidth=1.5)
    zoom_ax.plot(np.linspace(-250, 250, 501), Kernel_pred[j, :], c='#ff7f0e',dashes=(5,5), linestyle='--', linewidth=1.5)
    zoom_ax.plot(np.linspace(-250, 250, 501), Cond_upkernel[j, :, 250] - Kernel_pred[j, :], c='k', linestyle='-.', linewidth=1.5)
zoom_ax.set_xlim(-50, 50)
zoom_ax.set_ylim(-0.01, 0.05)
zoom_ax.set_xticks([-50, -25, 0, 25, 50])
zoom_ax.set_yticks([-0.0005, -0.00025, 0, 0.00025, 0.0005])
zoom_ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
zoom_ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
zoom_ax.patch.set_alpha(0) 
zoom_ax.axis('off')  
arrow_x = -30 
arrow_y =0.025
zoom_center_x = -110  
zoom_center_y =0.04
ax.annotate('', xy=(arrow_x, arrow_y), xytext=(zoom_center_x, zoom_center_y), arrowprops=dict(facecolor='#8B4513', edgecolor='#8B4513', arrowstyle='->', lw=2, mutation_scale=20))
plt.show()
plt.savefig(ResAnalyFile + '/Figure9_Theoretical_weight_results.png', dpi=300)



