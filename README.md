# Contador-de-celulas-blancas-con-redes-neuronales-convolucionals
Contador de células blancas utilizando redes neuronales convolucionales.

Este proyecto utilizará un modelo pre-entrenado (VGG16) utilizando
keras, el modelo proviene de [image-net.org](http://image-net.org/). 

El procedimiento seguido se hizo utilizando el siguiente [tutorial](https://www.learnopencv.com/keras-tutorial-transfer-learning-using-pre-trained-models/)

La idea es utilizar un modelo pre-entrenado utilizando [esta base de datos](http://image-net.org/synset?wnid=n05449959)

Una vez obtenido el modelo pre-entrenado se utiliza el modelo planteado en el siguiente repositorio [wbc-classification](https://github.com/dhruvp/wbc-classification/blob/master/notebooks/binary_training.ipynb). La idea es mejorar sus resultados ya que en ese modelo no se utilizan técnicas de convolución ya que el entrenamiento que utilizan fue entrenado en imágenes que rotaron levemente y se utilizará la misma matriz de confusión para medir qué tan efectivo es usar un modelo pre-entrenado. 

Después se utiliza crean cajas para enmarcar los resultados basándose en el siguiente [repositorio](https://github.com/fizyr/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb)

 
