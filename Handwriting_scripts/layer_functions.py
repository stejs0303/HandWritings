import keras.backend as K
import tensorflow as tf


def euclidian_distance(vectors):
    featuresA, featuresB = vectors
    
    sumSquared = K.sum(K.square(featuresA - featuresB), axis=1, keepdims=True)
    
    return K.sqrt(K.maximum(sumSquared, K.epsilon())) 


def contrastive_loss(y, preds, margin=1):
	y = tf.cast(y, preds.dtype)

	squaredPreds = K.square(preds)
	squaredMargin = K.square(K.maximum(margin - preds, 0))

	loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)

	return loss