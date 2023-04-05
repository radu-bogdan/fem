import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.constraints import Constraint

k1=49.4;k2=1.46;k3=520.6
f1=lambda x: k1/2/k2*(np.exp(k2*x[:,0]**2+k2*x[:,1]**2)-1)+1/2*k3*(x[:,0]**2+x[:,1]**2)

def getHandles(f=f1):
    elin=lambda x: x
    doubleexp=lambda x: tf.math.exp(x**2)
    exp=lambda x: tf.math.exp(x)
    def loss_fn(x,y):
      return(tf.math.reduce_sum((x-y)**2))

    
    class CustomConstraint (Constraint):
      def __init__(self, min=0.0001):
        self.ref_value = min
        
      def __call__(self, w):
        stabil=np.ones(w.shape)*1e-12
        wneu=tf.math.add(tf.math.multiply(w,tf.math.sign(w)),w)/2
        wneu=tf.math.add(wneu,stabil)
        return wneu
      def get_config(self):
        return {'ref_value': self.ref_value}
    
    model=Sequential()
    
    model.add(tf.keras.layers.InputLayer(input_shape=(2,)))
    model.add(Dense(10,activation=elin))
    model.add(Dense(100,activation=tf.nn.elu,kernel_constraint=keras.constraints.NonNeg(), kernel_regularizer=tf.keras.regularizers.L1(0.1)))
    model.add(Dense(100,activation=tf.nn.elu,kernel_constraint=keras.constraints.NonNeg(), kernel_regularizer=tf.keras.regularizers.L1(0.1)))
    model.add(Dense(100,activation=tf.nn.elu,kernel_constraint=keras.constraints.NonNeg(), kernel_regularizer=tf.keras.regularizers.L1(0.1)))
    model.add(Dense(10,activation=tf.nn.elu,kernel_constraint=keras.constraints.NonNeg(), kernel_regularizer=tf.keras.regularizers.L1(0.1)))
    model.add(Dense(10,activation=tf.nn.elu,kernel_constraint=keras.constraints.NonNeg(), kernel_regularizer=tf.keras.regularizers.L1(0.1)))
    model.add(Dense(1,activation=tf.nn.elu,kernel_constraint=keras.constraints.NonNeg()))
              
    model.compile(optimizer='adam', loss='MeanSquaredError',metrics=['accuracy'])
    
    xtrain=(np.random.rand(2,10000)-0.5)*4.2
    xtrain_tf=tf.constant(xtrain.T,dtype='float64')
    with tf.GradientTape() as tape:
      tape.watch(xtrain_tf)
      ytrain=f1(xtrain_tf)
            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
      grads = tape.gradient(ytrain,xtrain_tf)
    xtrain=xtrain.T
    ytrain=grads
    print(grads)
    center=f1(np.array([0,0]).reshape(1,2))
    optimizer = keras.optimizers.Adam()
    for k in range(10):
      epochs = 100
      xtrain=tf.constant(xtrain)
      for epoch in range(epochs):
          print("\nStart of epoch %d" % (epoch,))
    
          # Iterate over the batches of the dataset.
              # Open a GradientTape to record the operations run
              # during the forward pass, which enables auto-differentiation.
          with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape:
              tape.watch(xtrain)
                        # Run the forward pass of the layer.
                        # The operations that the layer applies
                        # to its inputs are going to be recorded
                        # on the GradientTape.
              logits = model(xtrain, training=True)  # Logits for this minibatch
                        # Compute the loss value for this minibatc
                    # Use the gradient tape to automatically retrieve
                    # the gradients of the trainable variables with respect to the loss.
              gradx = tape.gradient(logits,xtrain)
    
                  # Run the forward pass of the layer.
                  # The operations that the layer applies
                  # to its inputs are going to be recorded
                  # on the GradientTape.'  
                  # Compute the loss value for this minibatch.
            loss_value = loss_fn(ytrain, gradx)+np.linalg.norm(model(np.array([0,0]).reshape((1,2))-center))**2
              # Use the gradient tape to automatically retrieve
              # the gradients of the trainable variables with respect to the loss.
            grads = tape1.gradient(loss_value, model.trainable_weights)
              # Run one step of gradient descent by updating
              # the value of the variables to minimize the loss.
          optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
              # Log every 200 batches.
          if 1:
            print(
                      "Training loss (for one batch) at step %d: %.4f"
                % (epoch, float(loss_value))
                  )
            print("Seen so far: %s samples" % (epoch))
      x=np.zeros((2,200))
      x=x.reshape((200,2))
      xdiag=np.array(range(200))
      x[:,0]=xdiag/100
      yfun=f1(x)
      ymodel=model.predict(x)
      plt.plot(xdiag,ymodel,yfun)
      plt.pause(0.05)
            
    
    
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    Xm, Ym = np.meshgrid(x, y)
    Z=np.zeros(100*100)
    Z2=np.zeros(100*100)
    for k in range(100):
        for m in range(100): 
            Z[(k)*100+m]=model(np.array([k/100,m/100]).reshape(1,2))
            Z2[(k)*100+m]=f1(np.array([k/100,m/100]).reshape(1,2))
    
    Z2=Z2.reshape((100,100))
    Z=Z.reshape((100,100))
    plt.contour(Xm, Ym, Z, colors='black');
    plt.pause(0.05)
    plt.contour(Xm, Ym, Z2, colors='black');

    def fun(x):
        return model.predict(x)
    def  gradsandHessian(x):
        x = tf.constant(xtrain)
        with tf.GradientTape() as g2:
          with tf.GradientTape() as g1:
            g1.watch(x)
            g2.watch(x)
            with tf.GradientTape() as gg:
              gg.watch(x)
              y = model(x)
            G = gg.gradient(y, x)  # dy_dx = 2 * x
            H1 = g1.gradient(G[:,0], x)  # d2y_dx2 = 2
            H2 = g2.gradient(G[:,1], x)
        n=x.shape[0]
        H=np.zeros((n,2,2))
        H[:,0,:]=np.array(H1)
        H[:,1,:]=np.array(H2)
        G=np.array(G)
        return G,H
    return fun,gradsandHessian
