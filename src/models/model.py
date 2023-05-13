import tensorflow as tf
import tensorflow.keras.layers as layers
from src.models.trainertf_mod import Trainer, TrainerWithTest


class Embedding(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.in_shape = (224,224,3)             
        self.mnet_mdl = tf.keras.applications.MobileNetV2(input_shape=self.in_shape, include_top=False, weights='imagenet')
        for l in self.mnet_mdl.layers[:144]:
            l.trainable=False
        for l in self.mnet_mdl.layers[144:]:
            l.trainable=True
            
        self.global_ap = layers.GlobalAveragePooling2D()
        
        # Try and Error:
        #self.dense1 = layers.Dense(612, activation="relu")
        #self.norm1 = layers.BatchNormalization()

        # Create embeddings with Batch Normalization
        self.dense1 = layers.Dense(224, activation="relu")
        self.norm1 = layers.BatchNormalization()
        self.dense2 = layers.Dense(224, activation="softmax")
        # self.dense2 = layers.Dense(224, activation="sigmoid")


    def call(self, x0, training=False):
        x1 = self.mnet_mdl(x0)
        x2 = self.global_ap(x1)
        x3 = self.dense1(x2)
        x4 = self.norm1(x3)
        x5 = self.dense2(x4)
        return x5

class TripletLayer(layers.Layer):
    def __init__(self):
        super(TripletLayer, self).__init__()
    
    def call(self, X):
        anchor,positive,negative = X
        pos_distance = tf.math.reduce_sum(tf.math.square(anchor - positive), axis=-1)
        neg_distance = tf.math.reduce_sum(tf.math.square(anchor - negative), axis=-1)
        t_loss = tf.math.reduce_sum(tf.math.maximum(pos_distance - neg_distance + 0.2,0),axis=0)
        self.add_loss(t_loss)
        return t_loss

class FaceEmbeddings(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.in_shape = (224,224,3)
        self.embedding = Embedding()
        
    def call(self,X):
        #Embeddings for positive, anchor and negative
        anchor_input, positive_input, negative_input = X
        #print(positive_input)
        x_p = self.embedding(positive_input)
        x_a = self.embedding(anchor_input)
        x_n = self.embedding(negative_input)
        X_res = x_a , x_p , x_n
        return X_res

class MyModel(tf.keras.Model):
    def __init__(self, face_embedding):
        super().__init__()
        self.face_embeddings_mdl = face_embedding
        self.triplet_l = TripletLayer()
        self.x_los = 0
        
    def call(self, X):
        x1= self.face_embeddings_mdl(X)
        #Tripplet loss as layer
        self.x_los = self.triplet_l(x1)
        return x1
    
    def loss_fn(self):
        loss = self.x_los
        return loss
    

class MyTrainer(TrainerWithTest):
    
    def metrics(self, L, Y_pred):
        emb_a, emb_p, emb_n = Y_pred
        diff_1 = emb_p - emb_a
        diff_2 = emb_n - emb_a
        #Eucledian distance as metric
        sum_dif = tf.math.square(diff_1 - diff_2)
        acc = tf.math.sqrt(tf.math.reduce_sum(sum_dif))
        return {'loss': L, 'acc': acc}

    def train_report(self, epoch, step):
        print(f"    {epoch:03d}/{step:05d}:  loss {self.loss:6.4f}, acc (avg) {self.avg_acc:6.4f}")

    def test_report(self, epoch, step):
        print('-' * 50)
        print(f"> {epoch:03d}/{step + 1:05d}:  loss {self.test_loss:6.4f}")
        print('-' * 50)