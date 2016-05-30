
# coding: utf-8

# In[1]:

from pandas import read_csv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

get_ipython().magic(u'matplotlib inline')


# In[2]:

IMAGE_SIZE = 96


# Чтение данных (пока нужен только training.csv)

# In[4]:

#id_lookup = read_csv("../IdLookupTable.csv")
#sample_submission = read_csv("../SampleSubmission.csv")
#test = read_csv("../test.csv")
training = read_csv("../training.csv")


# Изображение записано в виде строки чисел от 0 до 255, разделенных пробелом. Преобразуем эту строку к массиву чисел 0-255

# In[5]:

training.Image = training.Image.apply(lambda img: np.fromstring(img, sep=' '))


# Отбрасывание примеров, в которых заполнены поля не для всех признаков. Из ~7000 изображение остается ~2000

# In[6]:

training = training.dropna()
#training = training.fillna(-1)


# labels - правильные координаты признаков, используемые для обучения сети

# In[7]:

labels = np.array([])
for key_name in training.columns.values[:-1]:
    labels = np.append(labels, np.array(training[key_name]))
labels = labels.reshape(training.shape[1]-1, len(training.Image)).T


# Для примера выведем на экран размерность массива признаков в виде (кол-во строк; кол-во признаков для одного изображения).
# И также выведем признаки для первого изображения

# In[8]:

print labels.shape
print labels[0]


# Функция для вывода изображения на экран

# In[9]:

def print_image(data):
    img = [[x]*3 for x in data]
    img = np.reshape(img, (96,96,3))
    plt.imshow(img)


# Функция для печати изображения с нанесенными признаками

# In[10]:

def test_point(img_ind, a, b):
    plt.xlim([0, 96])
    plt.ylim([96, 0])
    f = teX[img_ind].reshape(96*96)
    print_image(f)
    for i in range(length):
        plt.plot(a[img_ind][i][0],a[img_ind][i][1], 'r*')
        plt.plot(b[img_ind][i][0],b[img_ind][i][1], 'bo')
    plt.show()


# Функции для масштабирование признаков в прямую ((0; 96) -> (-1; 1)) и обратную сторону

# In[11]:

def scale(x):
    return (x - 48) / 48
    #return x / 96.0
def unscale(x):
    return x * 48 + 48
    #return x * 96.0


# Объявление массива образцов для обучения и тестирования, печать его размерности и первого элемента

# In[71]:

#t_samples = np.array([np.copy(img) for img in training.Image[0:1000].apply(lambda x: x / 255.0)])
t_samples = np.array([np.copy(img) for img in training.Image[:2048].apply(lambda x: x / 255.0)])
print t_samples.shape
print t_samples[0]


# Масштабирование признаков и печать первого элемента до масштабирования и после

# In[72]:

print labels[0]
t_labels = scale(labels)
print t_labels[0]


# Функция для инициализации весов (задаются размерности матриц (тензоров))

# In[14]:

def init_weights(shape):
    #return tf.Variable(tf.random_normal(shape, stddev=0.01))
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))


# Задание размеров обучающей и тестирующей выборки и вывод их на экран

# In[20]:

full_volume = len(t_samples)
train_volume = int(full_volume * 0.875)
test_volume = full_volume - train_volume
print "%d\n%d\n%d" % (full_volume, train_volume, test_volume)


# Задание обучающей и тестирующей выборки

# In[73]:

trX = np.array(t_samples[:train_volume], dtype=np.float)
teX = np.array(t_samples[train_volume:full_volume], dtype=np.float)
trY = np.array(t_labels[:train_volume], dtype=np.float)
teY = np.array(t_labels[train_volume:full_volume], dtype=np.float)

trX = trX.reshape(-1, 96, 96, 1)  # 96x96x1 input img
teX = teX.reshape(-1, 96, 96, 1)  # 96x96x1 input img


# Модель сверточной нейронной сети. p_keep_conv, p_keep_hidden - вероятности сохранения (не отбрасывания) нейрона при обучении после применения dropout

# In[84]:

def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden, show_shapes=False):
    
    l1 = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1 shape=(?, 96, 96, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1_shape = l1.get_shape()
    l1 = tf.nn.max_pool(l1, ksize=[1, 2, 2, 1],              # l1 shape=(?, 48, 48, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2 = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 48, 48, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2_shape = l2.get_shape()
    l2 = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1],              # l2 shape=(?, 24, 24, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)
    
    l3 = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 24, 24, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3_shape = l3.get_shape()
    l3 = tf.nn.max_pool(l3, ksize=[1, 3, 3, 1],              # l3 shape=(?, 8, 8, 128)
                        strides=[1, 3, 3, 1], padding='SAME')
    l3_after_maxpool_shape = l3.get_shape()
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    #l4 = tf.nn.tanh(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)
    
    #l5 = tf.nn.relu(tf.matmul(l4, w5))
    #l5 = tf.nn.dropout(l5, p_keep_hidden)

    y = tf.matmul(l4, w_o)
    #pyx = tf.nn.tanh(tf.matmul(l4, w_o))
    
    if show_shapes:
        print 'X: ', X.get_shape()
        print 'l1: ', l1_shape
        print 'l1 after max pool: ', l1.get_shape()
        print 'l2: ', l2_shape
        print 'l2 after max pool: ', l2.get_shape()
        print 'l3: ', l3_shape
        print 'l3 after max pool: ', l3_after_maxpool_shape
        print 'w4.get_shape().as_list()[0]: ', w4.get_shape().as_list()[0]
        print 'l3 after reshape: ', l3.get_shape()
        print 'l4: ', l4.get_shape()
        #print 'l5: ', l5.get_shape()
        print 'pyx: ', y.get_shape()
    return y


# Объявление placeholder'ов и инициализация весов

# In[85]:

X = tf.placeholder("float", [None, 96, 96, 1], name="X")
Y = tf.placeholder("float", [None, training.shape[1]-1], name="Y")

w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 8 * 8, 625]) # Fully connected 128 * 4 * 4 inputs, 625 outputs
#w5 = init_weights([2000, 625]) # FC 2000 input, 625 output
w_o = init_weights([625, training.shape[1]-1])         # FC 625 inputs, 2 outputs (labels)

p_keep_conv = tf.placeholder("float", name="p_keep_conv")
p_keep_hidden = tf.placeholder("float", name="p_keep_hidden")

pred = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden, show_shapes=False)


# Задание eps для проверки точности работы нейросети, а также размеров одновременно подаваемых образцов для обучения (они подаются нейросети, вычисляется средняя ошибка для всех этих образцов и только после этого меняются веса) и тестирования

# In[45]:

eps = 0.01
batch_size = 128


# После каждой эпохи обучения скорость обучения будет уменьшаться

# In[86]:

global_step = tf.Variable(1, trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step * batch_size,
                                           train_volume, 0.96, staircase=True)


# Задание вычислительных узлов графа tensorflow
# + Стоимость - среднеквадратичная ошибка
# + Метод обучения - RMSProp

# In[87]:

#cost = (tf.nn.l2_loss(py_x - Y))
cost = tf.reduce_mean(tf.reduce_sum(tf.square(pred - Y)))

train_op = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost, global_step=global_step)
#train_op = tf.train.AdamOptimizer().minimize(cost, global_step=global_step)

predict_op = pred


# Функции для оценки точности. Проверяется что все сравниваемые элементы отличаются не более чем на eps

# In[54]:

def eq_eps(y, yl):
    return np.abs(y - yl) < eps

def all_eq_eps(y, yl):
    return (all(eq_eps(yc, ylc)) for yc, ylc in zip(y, yl))


# Запуск обучения.
# * Создание сессии tensorflow
# * Инициализация переменных
# * Задание кол-ва эпох
# * Разделение обучающей выборки на равные части (batch) размером batch_size
# * Каждую эпоху:
#   - Для каждого batch
#     + Запустить обучение на этой части выборки
#   - Запустить сеть на тестирующей выборке
#   - Вычислить процент правильных результатов
#   - Напечатать результат

# In[63]:

sess = tf.Session()


# In[88]:

sess.run(tf.initialize_all_variables())

EPOCHS = 10

training_batch = zip(range(0, train_volume, batch_size),
                         range(batch_size, train_volume, batch_size))

for i in range(EPOCHS):
    for start, end in training_batch:
        
        feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_conv: 0.8, p_keep_hidden: 0.5}
        _, cst, cur_learning_rate = sess.run([train_op, cost, learning_rate], feed_dict=feed_dict)

    y = sess.run(predict_op, feed_dict={X: teX, p_keep_conv: 1.0, p_keep_hidden: 1.0})

    acc = [eq_eps(a,b) for (a,b) in zip(teY, y)]
    accuracy = np.mean(acc)
    print "Epoch: %d;   Cost: %7.3f;   Accuracy: %.0f%%;   Learning rate: %.5f" % (i, cst, accuracy*100, cur_learning_rate)


# Запустить сеть на первых 10 изображениях из тестового множества. И отмасштабировать к (0; 96).

# In[89]:

y = sess.run(predict_op, feed_dict={X: teX[0:10], p_keep_conv: 1.0, p_keep_hidden: 1.0})
a, b = unscale(y[0:10]), unscale(teY[0:10])


# Вывести на экран результат работы сети (a) и правильный результат (b) для одного и того же изображения

# In[90]:

print a[1]
print b[1]


# Изменить размеры 

# In[91]:

length = len(a[0]) / 2
a = a.reshape(10, length, 2)
b = b.reshape(10, length, 2)


# Вывести на экран изображения с нанесением признаков (синий - правильные, красный - результат сети)

# In[93]:

for i in range(10):
    test_point(i, a, b)


# Вывести веса первого сверточного слоя ради интереса

# In[94]:

print w.eval(session=sess)


# Закрытие сессии tensorflow

# In[95]:

sess.close()

