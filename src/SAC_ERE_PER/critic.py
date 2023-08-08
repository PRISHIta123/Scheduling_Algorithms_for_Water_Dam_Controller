import tensorflow as tf
tf.random.set_seed(1000)

#Critic (value) model

class Critic:

    def __init__(self, state_size, action_size, learning_rate):
        
        self.state_size= state_size
        self.action_size= action_size
        self.learning_rate= learning_rate
        self.value_target= self.critic_target()
        self.value= self.critic_value()
        self.Q1= self.critic_loss()
        self.Q2= self.critic_loss()

    def custom_loss(self, y_true, y_pred):

        loss = tf.keras.backend.mean(tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred), axis=1)) # need to put axis
        return loss

    #Critic Value function
    def critic_value(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(32,input_shape=(self.state_size,), activation = tf.keras.activations.relu,kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
        model.add(tf.keras.layers.Dense(32, activation = tf.keras.activations.relu,kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
        model.add(tf.keras.layers.Dense(1))
        optim = tf.keras.optimizers.Adam(lr = self.learning_rate)
        model.compile(loss = self.custom_loss, optimizer = optim)

        return model

    #Critic Target Value function
    def critic_target(self):
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(32,input_shape=(self.state_size,), activation = tf.keras.activations.relu,kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
        model.add(tf.keras.layers.Dense(32, activation = tf.keras.activations.relu,kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
        model.add(tf.keras.layers.Dense(1))
        optim = tf.keras.optimizers.Adam(lr = self.learning_rate)
        model.compile(loss = self.custom_loss, optimizer = optim)

        return model

    #Critic Loss function
    def critic_loss(self):
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(32,input_shape=(self.state_size+self.action_size,), activation = tf.keras.activations.relu,kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
        model.add(tf.keras.layers.Dense(32, activation = tf.keras.activations.relu,kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
        model.add(tf.keras.layers.Dense(1))
        optim = tf.keras.optimizers.Adam(lr = self.learning_rate)
        model.compile(loss = self.custom_loss, optimizer = optim)

        return model

        
