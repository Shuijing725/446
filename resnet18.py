import numpy as np
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Conv3D, BatchNormalization, Activation
from keras.layers import MaxPooling3D, AveragePooling3D, Flatten, Dropout
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

# load data from files
# length of train_label / train_feature = 4602
# length of valid_feature = 1971
train_label = np.load('../../data/train_binary_Y.npy')
train_feature = np.load('../../data/train_X.npy')
valid_feature = np.load('../../data/valid_test_X.npy')

# balance training data by oversampling
count = np.empty(len(train_label[0]))
for i in range(len(train_label[0])):
	count[i] = np.count_nonzero(train_label[:, i])

maximum = np.amax(count)
for i in range(len(count)):
	feature = train_feature[train_label[:, i] == 1]
	label = train_label[train_label[:, i] == 1]
	# print('sample:', len(label) * 1.0 / len(train_label))
	# print('actual count:', count[i] * 1.0 / maximum)
	# print('')
	ratio = int(maximum * 1.0 / count[i]) / 2
	# multiply the data containing minority features
	# print(np.tile(label, (ratio, 1, 1, 1)).shape)
	# print(train_label.shape)
	train_feature = np.concatenate((train_feature, np.tile(feature, (ratio, 1, 1, 1))), axis = 0)
	train_label = np.concatenate((train_label, np.tile(label, (ratio, 1))), axis = 0)

np.save('naive_oversample_train_feature', train_feature)
np.save('naive_oversample_train_label', train_label)


# normalize training data
# train_feature = (train_feature - np.mean(train_feature)) / np.std(train_feature)
num_train = len(train_label)
num_valid = len(valid_feature)

# define parameters
batch_size = 128
num_classes = 19
epochs = 45

width = 26
length = 31
height = 23

# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_valid = keras.utils.to_categorical(y_valid, num_classes)

# num_filters = 63

# Start model definition.
def make_model():
	num_filters = 64
	inputs = Input(shape=(width, length, height, 1))
	x = Conv3D(num_filters,
	           kernel_size=7,
	           padding='same',
	           strides=2,
	           kernel_initializer='he_normal',
	           kernel_regularizer=l2(1e-4))(inputs)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	# Orig paper uses max pool after 1st conv.
	# Reaches up 87% acc if use_max_pool = True.
	# Cifar10 images are already too small at 32x32 to be maxpooled. So, we skip.

	x = MaxPooling3D(pool_size=3, strides=1, padding='same')(x)
	# num_blocks = 3

	num_blocks = 4
	num_sub_blocks = 2
	# Instantiate convolutional base (stack of blocks).
	for i in range(num_blocks):
	    for j in range(num_sub_blocks):
	        strides = 1
	        is_first_layer_but_not_first_block = j == 0 and i > 0
	        if is_first_layer_but_not_first_block:
	            strides = 2
	        y = Conv3D(num_filters,
	                   kernel_size=3,
	                   padding='same',
	                   strides=strides,
	                   kernel_initializer='he_normal',
	                   kernel_regularizer=l2(1e-4))(x)
	        y = BatchNormalization()(y)
	        y = Activation('relu')(y)
	#        y = Conv3D(num_filters,
	#                   kernel_size=3,
	#                   padding='same',
	#                   kernel_initializer='he_normal',
	#                   kernel_regularizer=l2(1e-4))(y)
	#        y = BatchNormalization()(y)
	        if is_first_layer_but_not_first_block:
	            x = Conv3D(num_filters,
	                       kernel_size=1,
	                       padding='same',
	                       strides=2,
	                       kernel_initializer='he_normal',
	                       kernel_regularizer=l2(1e-4))(x)
	        x = keras.layers.add([x, y])
	        x = Activation('relu')(x)

	    num_filters = 2 * num_filters

	# Add classifier on top.
	x = AveragePooling3D()(x)
	y = Flatten()(x)
	outputs = Dense(num_classes,
	                activation='sigmoid',
	                kernel_initializer='he_normal')(y)

	model = Model(inputs = inputs, outputs = outputs)

	return model


# train/validation split ratio
split_ratio = 0.2
k_fold = 5

# store 5 models in an array
predictions = np.zeros((num_valid, num_classes))
preds = []
# five fold cross validation
for i in range(1):
	# define the start and end point of validation data
	start_point = int(num_train * split_ratio * i)
	end_point = start_point + int(num_train * split_ratio)

	# define train & validation data
	x_train = np.vstack((train_feature[0:start_point], train_feature[end_point:(num_train - 1)]))
	y_train = np.vstack((train_label[0:start_point], train_label[end_point:(num_train - 1)]))

	x_valid = train_feature[start_point:end_point]
	y_valid = train_label[start_point:end_point]

	x_train = x_train.reshape(len(x_train), width, length, height, 1)
	x_valid = x_valid.reshape(len(x_valid), width, length, height, 1)
	x_test = valid_feature.reshape(num_valid, width, length, height, 1)


	model = make_model()
	model.compile(loss=keras.losses.binary_crossentropy,
	              optimizer=SGD(lr=0.1, momentum = 0.9, nesterov = True),
	              metrics=['accuracy'])

	model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1,
	          shuffle = True,
	          validation_data=(x_valid, y_valid),
		  callbacks = [EarlyStopping(min_delta=0.1, patience=8)])

	# save model and weights
	model_json = model.to_json()
	with open("model" + str(i) + ".json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("weights" + str(i) + ".h5")
	print("iteration # ", i, ": Saved model to disk")

	# test model on validation data 
	predict = model.predict(x_test, batch_size = batch_size)
	predict[predict>=0.5] = 1
	predict[predict<0.5] = 0
	preds.append(predict)
	print(predict)
	predictions += predict

# majority voting among k models
low = predictions < 0.5# 3
predictions[low] = 0
high = predictions >= 0.5# 3
predictions[high] = 1

np.save('prediction', predictions)
# for i in range(k_fold):
# 	np.save('prediction' + str(i), preds[i])
