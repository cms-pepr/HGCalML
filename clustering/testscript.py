import tensorflow as tf
import setGPU
import numpy as np
import matplotlib.pyplot as plt
from object_condensation import object_condensation_loss
# from segmentation_sota import SpatialEmbLossTf
lovasz = False

n_points = 50

# lovasz_loss_calculator = SpatialEmbLossTf(clustering_dims=2)
class OverfittingKing(tf.keras.models.Model):
    def __init__(self, clustering_dims=2):
        super(OverfittingKing, self).__init__()
        n_units = 32
        self.layer_1 = tf.keras.layers.Dense(n_units, input_dim=2, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        
        self.layer_2 = tf.keras.layers.Dense(n_units, input_dim=n_points*n_units, activation='relu')
        self.layer_3 = tf.keras.layers.Dense(n_units, input_dim=n_units, activation='relu')
        self.layer_4 = tf.keras.layers.Dense(n_points*n_units, input_dim=n_units, activation='relu')
        
        self.reshape_unflat = tf.keras.layers.Reshape([n_points,n_units])
        self.layer_5 = tf.keras.layers.Dense(n_units, input_dim=n_units, activation='relu')
        self.layer_output_clustering_space = tf.keras.layers.Dense(clustering_dims, activation=None)
        self.layer_output_beta = tf.keras.layers.Dense(1, activation=None)
    def call(self, input_data):
        x = self.layer_1(input_data)
        x = self.flatten(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        #x = self.reshape_flat(x)
        x = self.layer_4(x)
        x = self.reshape_unflat(x)
        print('x',x.shape)
        x = self.layer_5(x)
        x1 = self.layer_output_clustering_space(x) * (1 if lovasz else 1000)
        x2 = tf.nn.sigmoid(self.layer_output_beta(x))
        return x1, tf.squeeze(x2, axis=-1)

n_clusters = 3
x_max = 1
y_max = 1
var = 0.02
np.random.seed(0)
data = []
cx = np.random.uniform(0, x_max, n_clusters)[np.newaxis, ...]
cy = np.random.uniform(0, y_max, n_clusters)[np.newaxis, ...]
px = np.random.normal(0, var, n_points)[..., np.newaxis]
py = np.random.normal(0, var, n_points)[..., np.newaxis]
classes = np.random.randint(0, n_clusters, n_points)
clusters = tf.one_hot(classes, n_clusters).numpy()
px = np.sum((px + cx)*clusters, axis=1)
py = np.sum((py + cy)*clusters, axis=1)
input_data = tf.concat((px[..., np.newaxis], py[..., np.newaxis]), axis=1)
input_data = tf.cast(input_data, tf.float32)
input_data = tf.expand_dims(input_data,axis=0)
print('input_data',input_data.shape)

my_model = OverfittingKing(clustering_dims=(4 if lovasz else 2))
my_model.call(input_data)
row_splits = tf.convert_to_tensor([0, n_points])
row_splits = tf.cast(row_splits, tf.int32)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
plt.scatter(px, py, s=2, c=classes, cmap=plt.get_cmap('Dark2'))
#exit()
plt.show()
def save_plot_to_file(it, clustering_values, classes, input, lovasz):
    plt.cla()
    if lovasz:
        clustering_values = tf.tanh(clustering_values)
        unit_displacement = tf.reduce_sum(tf.pow(clustering_space, 2), axis=-1)
        # clustering_values = clustering_values*0
        plt.scatter(clustering_values[:, 0].numpy() + input[:, 0].numpy(), clustering_values[:, 1].numpy() + input[:, 1].numpy(), s=2, c=classes,
                    cmap=plt.get_cmap('Dark2'))
    else:
        plt.scatter(clustering_values[:, 0].numpy(), clustering_values[:, 1].numpy(), s=2, c=classes,
                    cmap=plt.get_cmap('Dark2'))
    plt.xlabel('Clustering dimension 1')
    plt.ylabel('Clustering dimension 2')
    plt.savefig('images/%05d.png'%it)
for i in range(100000):
    with tf.GradientTape() as tape:
        clustering_space, beta_values = my_model.call(input_data)
        clustering_space, beta_values = clustering_space[0], beta_values[0] #get rid of '1' batch dimension 
        if lovasz:
            pass
            # loss = lovasz_loss_calculator(row_splits, input_data, clustering_space, beta_values, classes)
        else:
            loss, losses = object_condensation_loss(clustering_space, beta_values, classes, row_splits, Q_MIN=1, S_B=0.3)
        if i%100 == 0:
            print("Saving figure")
            save_plot_to_file(i, clustering_space, classes, input_data, lovasz)
        print(loss, losses)
    grads = tape.gradient(loss, my_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, my_model.trainable_variables))
