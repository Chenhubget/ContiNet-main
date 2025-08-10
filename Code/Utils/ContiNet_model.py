from Utils.Packages import *

# UpContiNet  
def UpContiNet_origin(Train_x,Train_y,Train_label):
    Dim_input = Train_x.shape[1]; Dim_label=Train_label.shape[1]; Dim_output=Train_y.shape[1]
    inp = Input(shape=(Dim_input,))
    inpt = inp
    lab = Input(shape=(Dim_label,)) 
    label = Dense(Dim_input*Dim_output,kernel_initializer='zeros',activation='LeakyReLU',use_bias=True)(lab)      
    label = Dense(Dim_input*Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_input*Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_input*Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_input*Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_input*Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_input*Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_input*Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_input*Dim_output,activation='linear')(label)
    if Train_y.shape[1] == 1:
        out = Dot(axes=1)([inpt,label])
    else:
        bbb = tf.reshape(inpt,shape=[-1,1,Dim_input])
        aaa=tf.reshape(label,shape=[-1,Dim_input,Dim_output])
        out = Dot(axes=[2,1])([bbb,aaa])
        out = tf.reshape(out,shape=[-1,Dim_output])
    output = out
    model =models.Model(inputs=[inp, lab], outputs=output)
    return model
 

# UpContiNet with single UC calculation point
def UpContiNet_sp(Train_x,Train_y,Train_label):
    Dim_input = Train_x.shape[1]; Dim_label=Train_label.shape[1]; Dim_output=Train_y.shape[1]
    inp = Input(shape=(Dim_input,))
    inpt = inp
    lab = Input(shape=(Dim_label,)) 
    label = Dense(Dim_input*Dim_output,kernel_initializer='zeros',activation='LeakyReLU',use_bias=True)(lab)     
    label = Dense(Dim_input*Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_input*Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_input*Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_input*Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_input*Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_input*Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_input*Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_input*Dim_output,activation='tanh')(label)  
    if Train_y.shape[1] == 1:
        out = Dot(axes=1)([inpt,label])
    else:
        bbb = tf.reshape(inpt,shape=[-1,1,Dim_input])
        aaa=tf.reshape(label,shape=[-1,Dim_input,Dim_output])
        out = Dot(axes=[2,1])([bbb,aaa])
        out = tf.reshape(out,shape=[-1,Dim_output])
    output = out
    model =models.Model(inputs=[inp, lab], outputs=output)
    return model
 
# UpContiNet with all UC calculation points
def UpContiNet_ap(Train_x,Train_y,Train_label):
    Dim_input = Train_x.shape[1]; Dim_label=Train_label.shape[1]; Dim_output=Train_y.shape[1]
    inp = Input(shape=(Dim_input,))
    inpt = inp
    lab = Input(shape=(Dim_label,)) 
    label = Dense(Dim_output,kernel_initializer='zeros',activation='LeakyReLU',use_bias=True)(lab)      
    label = Dense(Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_output,activation='tanh')(label)
    label_reshaped = Lambda(lambda x: tf.linalg.LinearOperatorToeplitz(x[0], x[1]).to_dense())([label, label])
    if Train_y.shape[1] == 1:
        out = Dot(axes=1)([inp, label_reshaped])
    else:
        inp_reshaped = tf.reshape(inp, shape=[-1, 1, Dim_input])
        out = Dot(axes=[2, 1])([inp_reshaped, label_reshaped])
        out = tf.reshape(out, shape=[-1, Dim_output])
    output = out
    model =models.Model(inputs=[inp, lab], outputs=output)
    return model

# DnContiNet
def DnContiNet(Train_x, Train_y, Train_label):
    Dim_input = Train_x.shape[1]
    Dim_label = Train_label.shape[1]
    Dim_output = Train_y.shape[1]
    inp = Input(shape=(Dim_input,))
    lab = Input(shape=(Dim_label,)) 
    label = Dense(Dim_output,kernel_initializer='zeros',activation='LeakyReLU',use_bias=True)(lab)     
    label = Dense(Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_output,activation='LeakyReLU')(label)
    label = Dense(Dim_output,activation='tanh')(label)
    label_reshaped = Lambda(lambda x: tf.linalg.LinearOperatorToeplitz(x[0], x[1]).to_dense())([label, label])
    if Dim_output == 1:
        out = Dot(axes=1)([inp, label_reshaped])
    else:
        inp_reshaped = tf.reshape(inp, shape=[-1, 1, Dim_input])
        out = Dot(axes=[2, 1])([inp_reshaped, label_reshaped])
        out = tf.reshape(out, shape=[-1, Dim_output])
    out_add = Dense(Dim_output, activation='tanh', kernel_initializer='zeros')(out)   
    out_add = Dense(Dim_output, activation='LeakyReLU')(out_add)
    out_add = Dense(Dim_output, activation='LeakyReLU')(out_add)
    out_add = Dense(Dim_output, activation='LeakyReLU')(out_add)
    out_add = Dense(Dim_output, activation='tanh')(out_add)
    out_add = Dense(Dim_output)(out_add)
    output = out   + lab * out_add     
    model = Model(inputs=[inp, lab], outputs=output)
    return model
