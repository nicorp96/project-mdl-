from turtle import ycor
import tensorflow as tf

class DataGenerator:
    def __init__(self, data_path_p,data_path_a,data_path_n, img_height, img_width, batch_size):
        self._img_height = img_height
        self._img_width = img_width
        self._batch_size = batch_size
        self._data_path_p = data_path_p
        self._data_path_a = data_path_a
        self._data_path_n = data_path_n

    def preprocess(self, x):
        #img = tf.image.random_brightness(x, max_delta=0.05)
        x = tf.keras.applications.mobilenet.preprocess_input(x)
        img = tf.image.resize(x, (224,224))
        return img

    def preprocess_anchor(self, x, y):
        #img = tf.image.random_brightness(x, max_delta=0.05)
        x = tf.keras.applications.mobilenet.preprocess_input(x)
        img = tf.image.resize(x, (224,224))
        return img, y

    def data_generator_test(self):
        ds_pos = tf.keras.preprocessing.image_dataset_from_directory(
            self._data_path_p,
            labels=None,
            color_mode='rgb',
            batch_size=self._batch_size,
            image_size=(self._img_height, self._img_width),
            shuffle=False,
            seed=123,
        )

        ds_anc = tf.keras.preprocessing.image_dataset_from_directory(
            self._data_path_a,
            labels= 'inferred',
            label_mode = 'binary',
            color_mode='rgb',
            batch_size=self._batch_size,
            image_size=(self._img_height, self._img_width),
            shuffle=False,
            seed=123,
        )

        ds_neg = tf.keras.preprocessing.image_dataset_from_directory(
            self._data_path_n,
            labels=None,
            color_mode='rgb',
            batch_size=self._batch_size,
            image_size=(self._img_height, self._img_width),
            shuffle=False,
            seed=123,
        )
        
        ds_pos = ds_pos.map(self.preprocess)
        ds_anc = ds_anc.map(self.preprocess_anchor)
        ds_neg = ds_neg.map(self.preprocess)
        
        return ds_pos, ds_anc, ds_neg

    def data_generator(self):
        ds_pos = tf.keras.preprocessing.image_dataset_from_directory(
            self._data_path_p,
            labels=None,
            color_mode='rgb',
            batch_size=self._batch_size,
            image_size=(self._img_height, self._img_width),
            shuffle=True,
            seed=123,
        )

        ds_anc = tf.keras.preprocessing.image_dataset_from_directory(
            self._data_path_a,
            labels= None,
            color_mode='rgb',
            batch_size=self._batch_size,
            image_size=(self._img_height, self._img_width),
            shuffle=True,
            seed=123,
        )

        ds_neg = tf.keras.preprocessing.image_dataset_from_directory(
            self._data_path_n,
            labels=None,
            color_mode='rgb',
            batch_size=self._batch_size,
            image_size=(self._img_height, self._img_width),
            shuffle=True,
            seed=123,
        )
        
        ds_pos = ds_pos.map(self.preprocess)
        ds_anc = ds_anc.map(self.preprocess)
        ds_neg = ds_neg.map(self.preprocess)
        
        return ds_pos, ds_anc, ds_neg

    def get_data_sets(self, test=False):
        if(test):
            ds_p, ds_a, ds_n = self.data_generator_test()
        else:
            ds_p, ds_a, ds_n = self.data_generator()
        dataset = tf.data.Dataset.zip((ds_p, ds_a, ds_n))
        dataset = dataset.shuffle(buffer_size=1024)
        return dataset
        

    def shape_of_mini_batches(self, dataset, number):
        for x in dataset.take(1):
            print('Image --> ', x.shape)
