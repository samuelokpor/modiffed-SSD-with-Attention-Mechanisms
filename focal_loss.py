import tensorflow as tf

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    # y_true: one-hot encoded ground truth labels
    # y_pred: predicted class probabilities
    # alpha: balancing parameter for positive and negative examples
    # gamma: focusing parameter to down-weight easy examples

    # convert one-hot encoded ground truth labels to class indices
    y_true = tf.argmax(y_true, axis=-1)

    # compute cross-entropy loss
    ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    # compute class probabilities from logits
    y_pred = tf.nn.softmax(y_pred, axis=-1)

    # compute weights based on the focal loss formula
    alpha_factor = tf.ones_like(y_true) * alpha
    alpha_factor = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_weight = alpha_factor * tf.pow(1 - pt, gamma)

    # compute the final loss by multiplying the cross-entropy loss with the weights
    focal_loss = focal_weight * ce_loss
    return tf.reduce_mean(focal_loss)