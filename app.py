import os
import numpy as np
import matplotlib.pyplot as plt



# Основные Шаги создания
# Типа Нейросети - Перцептрон (радиально базисные, сверточные и адаптивного резонанса пока не трогаем)
# Количество Эпох обучения. -  делаем циклом , немного проходов обучения (либо одного дата сета либо добавляя новые на каждом новом проходе)
# Количество Слоев - Слои соединены Синопсами(Весами) - числовыми матрицами (1-2 слой [4:5], 2-3 слой [3:4]).
# FORWARD PROPAGATION (hidden + output, движемся от начала к концу) - создание весов
# Тип инициализации Весов - Инициализация Весов Случайным образом лучше всего 
# Нейрон смещения = 1 (для графика функции активации)
# Функция активации (она же график) - Рилу | (лийнейная (f(x)=x, ReLU(max(0,x)) предпочтительная, Сигмойд сложная и не всегда эффективная)
# Функция потерь(разницы) - MSE (The Mean Squared Error) - считаем потери и точность нейросети между эпохами обучения
# BACK PROPAGATION (от output к hidden до самого первого hidden, движемся от конца к началу) - повторяется для каждого из скрытых слоев!
# --- корректировка (обучение) весов
# Первый Входной слой - по числу пикселей входящего изображения -  цвета пикселей из примера обучения (0 белый, 0,001-0,999 серый, 1 черный)
# Скрытые слои 
# Выходной слой - по числу вариантов ответа (если да нет и возможно то 3, если определение числа то кол-во чисел которые надо определить
# numpu.savez(file, *arg,**kwds) (numpy.load) - сохраняем результат обучение я если надо. Формат .npz


# Датасет обучения - к примеру mnist (mnist.npz)


# Определение папки исполняемого файла
#print(os.getcwd())
basedir = os.path.abspath(os.path.dirname(__file__))


# Загрузка дата сета для обучения, приведение его в нужный формат
def load_dataset():
	
	with np.load(basedir+"/mnist.npz") as f:
		
		# x_train - массив пикселей изображения из f (из mnist.npz)
        # convert from RGB to Unit RGB (перебрасывает в новый формат(в значения от 0 до 1 (так как только серые, а так их 255))
		x_train = f['x_train'].astype("float32") / 255

		# reshape from (60000, 28, 28) into (60000, 784) (меняет форму массива изображений с 60000 28Х28 на 60000 784 - в линию)
		# x_train - теперь имеет форму (num_images, image_width * image_height * 3)
    	# где num_images - количество изображений (в нашем случае 60 000 - это кол во изображений в mnist.npz)
    	# image_width * image_height * 3 - размер изображения без учета цвета
		# --- ВАЖНО умножение на 3 происходит если мы рассматриваем картинку в RGB цвете, при монохроме умножение на 3 не требуется.
		x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

		# labels, массив меток классов из f (из mnist.npz). Метки классов - это данные, которые показывают, к какому классу или категории 
		# принадлежит элемент. В контексте машинного обучения и обработки данных метки классов используются для определения того, 
		# к какой группе или категории данных относятся.
		# ---
		# Например, в задаче классификации изображений с цифрами 0-9, метки классов могут быть целочисленными значениями, 
		# которые показывают, к какой цифре относится изображение. В задаче классификации текста с использованием one-hot encoding, 
		# метки классов могут быть векторами с длиной, равной количеству классов, где каждый элемент равен 1 для соответствующего 
		# класса и нуля для других классов.
		y_train = f['y_train']

		# convert to output layer format (был одномерный массив из 60000 элем, стал многомерный массив 60000х10 так как будет 10 классов для 10 цифр)
        # --- для легкой корректировки Весов при обучении нейросети
		# np.eye(10) создает единичную матрицу размером 10x10, а затем используется индексация [y_train], чтобы преобразовать целочисленные 
		# метки классов в соответствующие векторы one-hot encoding. Например, если y_train содержит  метку класса 3(мы написали цифру 3), 
		# то соответствующий вектор one-hot encoding будет [1], где только третий элемент равен 1, а остальные равны 0.
		y_train = np.eye(10)[y_train]
        
		return x_train, y_train
	
images, labels = load_dataset()



# Инициализация матрицы Весов случайным образом (слой n, слой n-1). 
# --- Входящий слой     = числу пикселей входящего изображения
# --- Исходящий слой    = числу цифр (10) 
weights_input_to_hidden  = np.random.uniform(-0.5, 0.5, (20,784))
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10,20))

# Задаем нейрон смещения = 1 (Входящий слой(784) не учитывается)
bias_input_to_hidden = np.zeros((20,1)) 
bias_hidden_to_output = np.zeros((10,1)) 

# Эпохи обучения для корректировки весов при обучении
# - Важно не ставить высоких значений иначе нейронка будет не учится а запоминать!
epochs = 1

# Потери и Точность изначально = 0
e_loss = 0
e_correct = 0
# Рейтинг обучения
learning_rate = 0.01

for epoch in range(epochs):
	print(f'Epoch №{epoch}')

	# Приводим изображение и класс в форму двумерного массива (дальше он будет использоваться в перемножении матриц)
	# zip() создает кортеж из элементов с одинаковыми индексами из обоих списков,  [1,2] и [a,b], циклом получим  (1,a)(2,b)
	for image, label in zip(images, labels):

		# Создаем двумерные массивы
		image = np.reshape(image, (-1,1))
		label = np.reshape(label, (-1,1))

		# FORWARD PROPAGATION for hidden - первый этап обучения на скрытый слой
		# Передача данных через синопсы из Входящего слоя в Скрытый (нейрон смещения + переменная numpy умноженная на матрицу изображений)
		hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
		# Функция активации это нормализация нейронов (тут применяем сигмойд)
		# --- так как Значения нейронов изначально могут получится больше или меньше ожидаемого диапазона
		hidden = 1/(1+np.exp(-hidden_raw)) # sigmoid

		# FORWARD PROPAGATION for output - первый этап обучения на выходной слой
		output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
		output = 1/(1+np.exp(-output_raw)) # sigmoid

		# Подсчет насколько наше выхлоп нейросети отличается олт фактического значения (из lable), должна максимально к нему приближатся
		# Loss \ Errors calculating (плюсуем к уже существующему параметру(изначально 0) новые данные каждый проход цикла)
		e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
		e_correct += int(np.argmax(output) == np.argmax(label))

		# BACK PROPAGATION (output layer)
		delta_output = output - label
		weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
		bias_hidden_to_output += -learning_rate * delta_output

		# BACK PROPAGATION  (hidden layer)
		delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
		weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
		bias_input_to_hidden += -learning_rate * delta_hidden

	# for tqdm  lib
	#print(output)
	print(f'Loss: {round((e_loss[0] / images.shape[0]) * 100, 3)}%')
	print(f'Accuracy: {round((e_correct / images.shape[0]) * 100, 3)}%')
	# Обнуление после эпохи обучения (что бы новая эпоха начинала с чистого листа но уже кое что зная)
	e_loss = 0
	e_correct = 0


# CHECK CUSTOM - проверка 
test_image = plt.imread(basedir+"/custom.jpg", format="jpeg")

# Grayscale + Unit RGB + inverse colors (инверсия для черных цифр на белом фоне)
gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
test_image = 1 - (gray(test_image).astype("float32") / 255)

# Reshape
test_image = np.reshape(test_image, (test_image.shape[0] * test_image.shape[1]))

# Predict
image = np.reshape(test_image, (-1, 1))

# Forward propagation (to hidden layer)
hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
hidden = 1 / (1 + np.exp(-hidden_raw)) # sigmoid
# Forward propagation (to output layer)
output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
output = 1 / (1 + np.exp(-output_raw))

max_index = np.argmax(output)  # находим индекс максимального элемента
arr_without_max = np.delete(output, max_index)  # удаляем максимальный элемент из массива
second_max_index = np.argmax(arr_without_max) # находим индекс второго по величине значения элемента

[print(i,'-',output[i]) for i in range(len(output))]
print(max_index, output[max_index])
print(second_max_index, output[second_max_index])

# Так как обучение небольшое может быть погрешность потому пусть выводит наивысшее и второе возможное значение
plt.imshow(test_image.reshape(28, 28), cmap="Greys")
plt.title(f"NN suggests the CUSTOM number is: {output}, {max_index} or {second_max_index+1}")
plt.show()


