from Utils.Packages import *
from Utils.Auxiliary import *
from Utils.ContiNet_model import *

## Load data
parent_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)).replace('\\', '/')
InitialModelFile =  parent_directory + '/Results_1DiscreteUC_1DataSet_Evaluation/'
ResAnalyFile = parent_directory +'/Results_1DiscreteUC_2Generalization/'
Cond_upkernel_1to40 = np.array(h5py.File(parent_directory + '/Data_1UTrainSet4_Part1/' + 'Cond_upkernel_1to40.mat')['Cond_upkernel']).T
Cond_upkernel_2to38 = np.array(h5py.File(parent_directory + '/Data_1UTrainSet4_Part1/' + 'Cond_upkernel_2to38.mat')['Cond_upkernel']).T

# Visualize the results
models = [InitialModelFile + 'UC_Discrete_Modelparam_UTrainSet4_Model_Layers9_CalPNum1.h5', ResAnalyFile + 'UC_Discrete_Modelparam_UTrainSet4_Finetune2to38_Model_Layers9_CalPNum1.h5']
Colors = ['#ff7f0e','#2ca02c','#2ca02c','#ff7f0e']
plt.figure(figsize=(13,8.5))
plt.subplots_adjust(top=0.945,bottom=0.095,left=0.090,right=0.975,hspace=0.330,wspace=0.250)  
for i in range(2):
    model = load_model(models[i],custom_objects={'R2':R2})
    Midfiture_label = Model(inputs=model.input,outputs=model.get_layer('dense_8').output)
    Upkernel_1to40pred = Midfiture_label.predict([np.ones([14,501,1]),np.linspace(1,40,14)])
    Upkernel_2to38pred = Midfiture_label.predict([np.ones([13,501,1]),np.linspace(2,38,13)])
    Kernel_pred = [Upkernel_1to40pred,Upkernel_2to38pred]
    Kernel = [Cond_upkernel_1to40,Cond_upkernel_2to38]
    for j in range(2):
        plt.subplot(2,2,j+1+i*2)
        if i==0:
            index = j
        else:
            index =1-j
        for k in range(np.shape(Kernel[index])[0]):
            plt.plot(np.linspace(-250,250,501),Kernel[index][k,:,250],label='Weight_true',c='b',linewidth=1.5)
            plt.plot(np.linspace(-250,250,501),Kernel_pred[index][k,:],label='Weight_pred',ls='--',c=Colors[j+i*2],alpha=1,dashes=(5, 5), linewidth=1.5)
            plt.plot(np.linspace(-250,250,501),Kernel[index][k,:,250]-Kernel_pred[index][k,:],ls='-.',color='k',linewidth=1.5,label='Weight_error')
        plt.text(0.01,0.91,'('+chr(96+j+1+i*2)+')',transform=plt.gca().transAxes,fontsize=22,weight='bold')
        plt.ylabel('Weight',fontsize=23)
        plt.xlabel('Points_obs',fontsize=23);  
        plt.xticks([-250,-150,-50,0,50,150,250],fontsize=23)  
        Minn = np.min([np.min(Kernel[index][:,:,250]),np.min(Kernel_pred[index]),np.min(Kernel[index][:,:,250] - Kernel_pred[index])])
        Maxx = np.max([np.max(Kernel[index][:,:,250]),np.max(Kernel_pred[index]),np.max(Kernel[index][:,:,250] - Kernel_pred[index])])
        plt.yticks(np.linspace(Minn,Maxx,7),fontsize=23)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  
        plt.legend(['Weight_discrete','Weight_pred','Weight_error'],fontsize=18,loc='upper right',bbox_to_anchor=(1.022, 1.04),handletextpad=0.1,labelspacing=0.3,frameon=False)
        plt.xlim(-250,250)
        ax=plt.gca()
        ypad = [0.04,0.125,0.042,.06][j+i*2]   
        ygao = [0.27,0.21,0.27,0.27][j+i*2]   
        xpad =[0.0032,0.003,0.0025,0.003][j+i*2]  
        zoom_ax = plt.axes([plt.gca().get_position().x0  +xpad, plt.gca().get_position().y0 + ypad, 0.188,ygao])   
        for k in range(np.shape(Kernel[index])[0]):
            zoom_ax.plot(np.linspace(-50, 50, 101), Kernel[index][k,200:301,250],c='b',linewidth=1.5)
            zoom_ax.plot(np.linspace(-50, 50, 101), Kernel_pred[index][k,200:301],ls='--', c=Colors[j+i*2],alpha=1,dashes=(5, 5), linewidth=1.5)
            zoom_ax.plot(np.linspace(-50, 50, 101), Kernel[index][k,200:301,250] - Kernel_pred[index][k,200:301],ls='-.',color='k',linewidth=1.5,label='Weight_error')
        zoom_ax.set_xlim(-50, 50)
        zoom_ax.set_ylim(-0.008, 0.05)
        zoom_ax.tick_params(axis='both', which='both', labelsize=0, length=0)
        zoom_ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
        zoom_ax.patch.set_alpha(0)
        zoom_ax.axis('off')  
        zoom_center_x = -110  
        zoom_center_y = [0.04,0.04,0.022,0.042][j+i*2]   
        arrow_x = -30 
        arrow_y =[0.025,0.025,0.014,0.025][j+i*2] 
        ax.annotate('', xy=(arrow_x, arrow_y), xytext=(zoom_center_x, zoom_center_y), arrowprops=dict(facecolor='#8B4513', edgecolor='#8B4513', arrowstyle='->', lw=2, mutation_scale=20))
plt.show()
plt.savefig(ResAnalyFile + '/Figure7_Generalization.png',dpi=300)




 