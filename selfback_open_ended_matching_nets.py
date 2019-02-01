import numpy as np
from keras import backend as K
from keras.layers import Input, Lambda, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from scipy import fftpack
import keras
import os
import csv
import tensorflow as tf
from keras.layers.merge import _Merge

np.random.seed(1)

imus = [1, 2]

activityType = ["jogging", "sitting", "standing", "walkfast", "walkmod", "walkslow", "upstairs", "downstairs", "lying"]
idList = range(len(activityType))
activityIdDict = dict(zip(activityType, idList))


def write_data(file_path, data):
    if (os.path.isfile(file_path)):
        f = open(file_path, 'a')
        f.write(data + '\n')
    else:
        f = open(file_path, 'w')
        f.write(data + '\n')
    f.close()


class MatchCosine(_Merge):
    def __init__(self, nway=5, n_samp=1, **kwargs):
        super(MatchCosine, self).__init__(**kwargs)
        self.eps = 1e-10
        self.nway = nway
        self.n_samp = n_samp

    def build(self, input_shape):
        print('here')

    def call(self, inputs):
        self.nway = (len(inputs) - 2) / self.n_samp
        similarities = []

        targetembedding = inputs[-2]
        numsupportset = len(inputs) - 2
        for ii in range(numsupportset):
            supportembedding = inputs[ii]

            sum_support = tf.reduce_sum(tf.square(supportembedding), 1, keep_dims=True)
            supportmagnitude = tf.rsqrt(tf.clip_by_value(sum_support, self.eps, float("inf")))

            sum_query = tf.reduce_sum(tf.square(targetembedding), 1, keep_dims=True)
            querymagnitude = tf.rsqrt(tf.clip_by_value(sum_query, self.eps, float("inf")))

            dot_product = tf.matmul(tf.expand_dims(targetembedding, 1), tf.expand_dims(supportembedding, 2))
            dot_product = tf.squeeze(dot_product, [1])

            cosine_similarity = dot_product * supportmagnitude * querymagnitude
            similarities.append(cosine_similarity)

        similarities = tf.concat(axis=1, values=similarities)
        softmax_similarities = tf.nn.softmax(similarities)
        preds = tf.squeeze(tf.matmul(tf.expand_dims(softmax_similarities, 1), inputs[-1]))

        preds.set_shape((inputs[0].shape[0], self.nway))
        return preds

    def compute_output_shape(self, input_shape):
        input_shapes = input_shape
        return input_shapes[0][0], self.nway


def read_data(path):
    person_data = {}
    files = os.listdir(path)
    for f in files:
        temp = f.split("_")
        user = temp[0]
        activity = temp[1]
        data = []
        reader = csv.reader(open(os.path.join(path, f), "r"), delimiter=",")
        for row in reader:
            data.append(row)

        activity_data = {}
        if user in person_data:
            activity_data = person_data[user]
            activity_data[activity] = data
        else:
            activity_data[activity] = data
        person_data[user] = activity_data

    return person_data


def remove_class(_data, remove_classes):
    data = {}
    for user_id, labels in _data.items():
        _labels = {}
        for label in labels:
            if label not in remove_classes:
                _labels[label] = labels[label]
        data[user_id] = _labels
    return data


def extract_features(data, dct_length, win_len=500):
    people = {}
    for person in data:
        person_data = data[person]
        classes = {}
        for activity in person_data:
            df = person_data[activity]
            wts = split_windows(df, win_len, overlap_ratio=1)
            dct_wts = dct(wts, comps=dct_length)
            classes[activity] = dct_wts
        people[person] = classes
    return people


def split_windows(data, window_length, overlap_ratio=None):
    outputs = []
    i = 0
    N = len(data)
    increment = int(window_length * overlap_ratio)
    while i + window_length < N:
        start = i
        end = start + window_length
        outs = [a[:] for a in data[start:end]]
        i = int(i + (increment))
        outputs.append(outs)
    return outputs


def dct(windows, comps=60):
    dct_window = []
    for tw in windows:
        all_acc_dcts = np.array([])
        for index in imus:
            _index = index - 1
            x = [t[(_index * 3) + 0] for t in tw]
            y = [t[(_index * 3) + 1] for t in tw]
            z = [t[(_index * 3) + 2] for t in tw]

            dct_x = np.abs(fftpack.dct(x, norm='ortho'))
            dct_y = np.abs(fftpack.dct(y, norm='ortho'))
            dct_z = np.abs(fftpack.dct(z, norm='ortho'))

            v = np.array([])
            v = np.concatenate((v, dct_x[:comps]))
            v = np.concatenate((v, dct_y[:comps]))
            v = np.concatenate((v, dct_z[:comps]))
            all_acc_dcts = np.concatenate((all_acc_dcts, v))

        dct_window.append(all_acc_dcts)
    return dct_window


def support_set_split(_data, k_shot):
    support_set = {}
    everything_else = {}
    for user, labels in _data.items():
        _support_set = {}
        _everything_else = {}
        for label, data in labels.items():
            supportset_indexes = np.random.choice(range(len(data)), k_shot, False)
            supportset = [d for index, d in enumerate(data) if index in supportset_indexes]
            everythingelse = [d for index, d in enumerate(data) if index not in supportset_indexes]
            _support_set[label] = supportset
            _everything_else[label] = everythingelse
        support_set[user] = _support_set
        everything_else[user] = _everything_else
    return support_set, everything_else


def packslice(data_set, classes_per_set, samples_per_class, numsamples, train_classes, feature_length):
    n_samples = samples_per_class * classes_per_set
    support_cacheX = []
    support_cacheY = []
    target_cacheY = []

    ids = range(len(train_classes))
    classDict = dict(zip(train_classes, ids))

    for itr in range(numsamples):
        slice_x = np.zeros((n_samples + 1, feature_length))
        slice_y = np.zeros((n_samples,))

        ind = 0
        pinds = np.random.permutation(n_samples)

        x_hat_class = np.random.randint(classes_per_set)

        for j, cur_class in enumerate(train_classes):
            data_pack = data_set[cur_class]
            example_inds = np.random.choice(len(data_pack), samples_per_class, False)

            for eind in example_inds:
                slice_x[pinds[ind], :] = data_pack[eind]
                slice_y[pinds[ind]] = classDict[cur_class]
                ind += 1

            if j == x_hat_class:
                target_indx = np.random.choice(len(data_pack))
                while target_indx in example_inds:
                    target_indx = np.random.choice(len(data_pack))
                slice_x[n_samples, :] = data_pack[target_indx]
                target_y = classDict[cur_class]

        support_cacheX.append(slice_x)
        support_cacheY.append(keras.utils.to_categorical(slice_y, classes_per_set))
        target_cacheY.append(keras.utils.to_categorical(target_y, classes_per_set))

    return np.array(support_cacheX), np.array(support_cacheY), np.array(target_cacheY)


def create_train_instances(train_sets, classes_per_set, samples_per_class, train_size, train_classes, feature_length):
    support_X = None
    support_y = None
    target_y = None
    for user_id, train_feats in train_sets.items():
        _support_X, _support_y, _target_y = packslice(train_feats, classes_per_set, samples_per_class, train_size,
                                                      train_classes, feature_length)

        if support_X is not None:
            support_X = np.concatenate((support_X, _support_X))
            support_y = np.concatenate((support_y, _support_y))
            target_y = np.concatenate((target_y, _target_y))
        else:
            support_X = _support_X
            support_y = _support_y
            target_y = _target_y

    print("Data shapes: ")
    print(support_X.shape)
    print(support_y.shape)
    print(target_y.shape)
    return [support_X, support_y, target_y]


def packslice_test(data_set, support_set, samples_per_class, train_labels, feature_length):
    support_cacheX = []
    support_cacheY = []
    target_cacheY = []

    for _class in data_set:
        support_labels = []
        support_labels.extend(train_labels)
        for item in list(data_set.keys()):
            if item not in support_labels:
                support_labels.append(item)
        n_samples = samples_per_class * len(support_labels)

        ids = range(len(support_labels))
        classDict = dict(zip(support_labels, ids))

        support_X = np.zeros((n_samples, feature_length))
        support_y = []

        for i, _class_ in enumerate([f for f in support_set.keys() if f in support_labels]):
            _X = support_set[_class_]
            for j in range(len(_X)):
                support_X[(i * samples_per_class) + j, :] = _X[j]
                support_y.append(classDict[_class_])

        X = data_set[_class]
        y = classDict[_class]

        for index in range(len(X)):
            slice_x = np.zeros((n_samples + 1, feature_length))
            slice_y = []

            slice_x[:n_samples, :] = support_X
            slice_x[n_samples, :] = X[index]

            slice_y.extend(support_y)

            target_y = y

            support_cacheX.append(slice_x)
            support_cacheY.append(keras.utils.to_categorical(slice_y, len(support_labels)))
            target_cacheY.append(keras.utils.to_categorical(target_y, len(support_labels)))

    return np.array(support_cacheX), np.array(support_cacheY), np.array(target_cacheY)


def create_test_instance(test_set, support_set, samples_per_class, train_labels, feature_length):
    support_X = None
    support_y = None
    target_y = None

    for user_id, test_data in test_set.items():
        support_data = support_set[user_id]
        _support_X, _support_y, _target_y = packslice_test(test_data, support_data, samples_per_class, train_labels,
                                                           feature_length)

        if support_X is not None:
            support_X = np.concatenate((support_X, _support_X))
            support_y = np.concatenate((support_y, _support_y))
            target_y = np.concatenate((target_y, _target_y))
        else:
            support_X = _support_X
            support_y = _support_y
            target_y = _target_y

    print("Data shapes: ")
    print(support_X.shape)
    print(support_y.shape)
    print(target_y.shape)
    return [support_X, support_y, target_y]


def mlp_embedding(x):
    x = Dense(1200, activation='relu')(x)
    x = BatchNormalization()(x)
    return x


def holdout_train_test_split(user_data, test_ids):
    train_data = {key: value for key, value in user_data.items() if key not in test_ids}
    test_data = {key: value for key, value in user_data.items() if key in test_ids}
    return train_data, test_data


def get_hold_out_users(users):
    indices = np.random.choice(len(users), int(len(users) / 3), False)
    test_users = [u for indd, u in enumerate(users) if indd in indices]
    return test_users


def get_test_classes(all_clsses, test_lngth):
    indices = np.random.choice(len(all_clsses), test_lngth, False)
    test_clss = [u for indd, u in enumerate(all_clsses) if indd in indices]
    return test_clss


bsize = 128
samples_per_class = 5
train_size = 500
epochs = 10
dct_length = 60

classes = ["jogging", "sitting", "standing", "walkfast", "walkmod", "walkslow", "upstairs", "downstairs", "lying"]
test_classes = get_test_classes(classes, 1)

write_file = ''
data_path = ''

user_data = read_data(data_path)
feature_data = extract_features(user_data, dct_length)
user_data = {}
test_user_ids = get_hold_out_users(list(feature_data.keys()))

print(test_user_ids)
print('_'.join(test_classes))

_train_features, _test_features = holdout_train_test_split(feature_data, test_user_ids)

train_classes = [cls for cls in classes if cls not in test_classes]
feature_length = dct_length * 3 * 2

_train_features = remove_class(_train_features, test_classes)
train_data = create_train_instances(_train_features, len(train_classes), samples_per_class, train_size, train_classes,
                                    feature_length)

test_support_set, _test_features = support_set_split(_test_features, samples_per_class)
_test_features = remove_class(_test_features, train_classes)

test_data = create_test_instance(_test_features, test_support_set, samples_per_class, train_classes, feature_length)

model = None
y_pred_p = None
y_true_p = None
numsupportset = samples_per_class * len(train_classes)
input1 = Input((numsupportset + 1, feature_length))

modelinputs = []
for lidx in range(numsupportset):
    modelinputs.append(mlp_embedding(Lambda(lambda x: x[:, lidx, :])(input1)))
targetembedding = mlp_embedding(Lambda(lambda x: x[:, -1, :])(input1))
modelinputs.append(targetembedding)
supportlabels = Input((numsupportset, len(train_classes)))
modelinputs.append(supportlabels)

knnsimilarity = MatchCosine(nway=len(train_classes), n_samp=samples_per_class)(modelinputs)

model = Model(inputs=[input1, supportlabels], outputs=knnsimilarity)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit([train_data[0], train_data[1]], train_data[2], epochs=epochs, batch_size=bsize, verbose=1)

model_output = model.layers[83].output
model_encoder = K.function([model.input[0], model.input[1], K.learning_phase()], [model_output])

eps = 1e-10
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    for items, lbs, y_true in zip(test_data[0], test_data[1], test_data[2]):
        test_X = np.zeros((1, len(train_classes) * samples_per_class + 1, feature_length))
        test_X[:, 0, :] = items[-1]
        test_Y = np.zeros((1, len(train_classes) * samples_per_class, len(train_classes)))
        target_embed = model_encoder([test_X, test_Y, 0])[0]
        # print(target_embed)
        similarities = []

        for i in range(len(items) - 1):
            test_X = np.zeros((1, len(train_classes) * samples_per_class + 1, feature_length))
            test_X[:, 0, :] = items[i]
            support_embed = model_encoder([test_X, test_Y, 0])[0]
            # print(support_embed)
            sum_support = tf.reduce_sum(tf.square(support_embed), 1, keep_dims=True)
            supportmagnitude = tf.rsqrt(tf.clip_by_value(sum_support, eps, float("inf")))

            sum_query = tf.reduce_sum(tf.square(target_embed), 1, keep_dims=True)
            querymagnitude = tf.rsqrt(tf.clip_by_value(sum_query, eps, float("inf")))

            dot_product = tf.matmul(tf.expand_dims(target_embed, 1), tf.expand_dims(support_embed, 2))
            dot_product = tf.squeeze(dot_product, [1])

            cosine_similarity = dot_product * supportmagnitude * querymagnitude
            similarities.append(cosine_similarity)
        similarities = tf.concat(axis=1, values=similarities)
        softmax_similarities = tf.nn.softmax(similarities)
        test_Y = np.zeros((1, lbs.shape[0], lbs.shape[1]), dtype=np.float32)
        test_Y[0, :, :] = lbs
        preds = tf.squeeze(tf.matmul(tf.expand_dims(softmax_similarities, 1), test_Y))
        new_preds = sess.run(preds)
        y_pred = np.argmax(new_preds)
        y_true = np.argmax([y_true], axis=1)[0]
        write_data(write_file, str(y_pred) + ',' + str(y_true))
