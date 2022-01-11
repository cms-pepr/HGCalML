import tensorflow as tf
import numpy as np


print("MODULE OBSOLETE?",__name__)
raise ImportError("MODULE",__name__,"will be removed")

def lovasz_grad_x(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = tf.reduce_sum(gt_sorted)

    intersection = gts - tf.math.cumsum(gt_sorted)
    union = gts + tf.math.cumsum(1 - gt_sorted)
    jaccard = 1. - intersection / union


    if p > 1:  # cover 1-pixel case
        jaccard_2 = jaccard[1:p] - jaccard[0:-1]

    jaccard = tf.concat([[jaccard[0]], jaccard_2], axis=0)

    return jaccard


def lovasz_hinge_flat_x(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    logits = tf.cast(logits, tf.float32)
    labels = tf.cast(labels, tf.float32)

    signs = 2. * labels - 1.
    errors = (1. - logits * signs)


    perm = tf.argsort(errors, axis=0, direction='DESCENDING')[..., tf.newaxis]

    errors_sorted = tf.gather_nd(errors, perm)
    gt_sorted = tf.gather_nd(labels, perm)


    grad = lovasz_grad_x(gt_sorted)
    loss = tf.tensordot(tf.nn.relu(errors_sorted), grad, 1)

    return loss



def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        0/0
        #l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc

    return acc / n


def flatten_binary_scores_x(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]

    return vscores, vlabels

def lovasz_hinge_x(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """


    logits = tf.transpose(logits, (1,0))
    labels = tf.transpose(labels, (1,0))


    # print("TF", tf.reduce_sum(logits))


    if per_image:
        loss = mean(lovasz_hinge_flat_x(*flatten_binary_scores_x(tf.expand_dims(log, axis=0), tf.expand_dims(lab, axis=0), ignore))
                    for log, lab in zip(logits, labels))
    else:
        0/0
        # loss = lovasz_hinge_flat(
        #     *flatten_binary_scores(logits, labels, ignore))
    return loss

class SpatialEmbLossTf(tf.keras.layers.Layer):

    def __init__(self, to_center=True, n_sigma=1, foreground_weight=1,**kwargs):
        super(SpatialEmbLossTf, self).__init__(**kwargs)

        print('Created spatial emb loss function with: to_center: {}, n_sigma: {}, foreground_weight: {}'.format(
            to_center, n_sigma, foreground_weight))

        self.to_center = to_center
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight

        # # coordinate map
        # xm = tf.tile(tf.reshape(tf.linspace(0.,2.,2048), [-1,1,1]), [1,1024,1])
        # ym = tf.tile(tf.reshape(tf.linspace(0.,1.,1024), [1,-1,1]), [2048,1,1])
        # xym = tf.concat((xm,ym), axis=-1)
        # xym = tf.transpose(xym, [1,0,2])
        # self.xym = xym

        self.min_value = tf.convert_to_tensor([-4., -4., -600])[tf.newaxis, ...]
        self.max_value = tf.convert_to_tensor([4, 4, +600.])[tf.newaxis, ...]


        self.multiplier = tf.convert_to_tensor([100, 100, 1.])[tf.newaxis, ...]

    def call(self, row_splits, input, prediction, beta_values, classes, w_inst=1, w_var=10, w_seed=1, iou=False, iou_meter=None):
        classes = classes + 1

        # xym_s = self.xym[0:height, 0:width,:]
        loss = 0


        batch_size = len(row_splits) - 1

        for b in range(0, batch_size):
            prediction_s = prediction[row_splits[b]:row_splits[b+1]]
            input_s = input[row_splits[b]:row_splits[b+1]]
            classes_s = classes[row_splits[b]:row_splits[b+1]]
            num_vertices = row_splits[b+1] - row_splits[b]

            input_s_spatial = tf.concat((input_s[:,1][..., tf.newaxis],input_s[:,2][..., tf.newaxis],input_s[:,7][..., tf.newaxis]), axis=-1)

            input_s_spatial = (input_s_spatial - self.min_value)/ (self.max_value - self.min_value)
            # input_s_spatial = input_s_spatial - self.min_value

            # TODO: Make sure activations are correct
            spatial_emb = tf.math.tanh(prediction_s[:, 0:3]) + input_s_spatial
            sigma = prediction_s[:, 3:3+self.n_sigma]
            seed_map = tf.math.sigmoid(prediction_s[:, 3+self.n_sigma:3+self.n_sigma+1])

            # loss accumulators
            var_loss = 0
            instance_loss = 0
            seed_loss = tf.constant(0., tf.float32)
            obj_count = 0

            # instance = tf.expand_dims(instances[b], axis=-1)
            # label = tf.expand_dims(labels[b], axis=-1)

            instance_ids,_ = tf.unique(tf.reshape(classes_s, (-1,)))
            instance_ids = instance_ids[instance_ids!=0]
            instance_ids = tf.sort(instance_ids)

            # regress bg to zero
            bg_mask = classes == 0
            if tf.reduce_sum(tf.cast(bg_mask, tf.int32)) > 0:
                seed_loss += tf.reduce_sum(tf.math.pow(seed_map[bg_mask]-0.,2.))

            for id in instance_ids:
                in_mask = classes_s == id
                if self.to_center:
                    center = tf.reduce_mean(input_s_spatial[in_mask], axis=0)
                else:
                    print("Not implemented yet")
                    0/0
                    center = tf.reshape(tf.reduce_mean(tf.reshape(spatial_emb[tf.broadcast_to(in_mask,spatial_emb.shape)], [-1, 2]), axis=0), [1,1,2])

                # calculate sigma
                sigma_in = sigma[in_mask]

                s = tf.reduce_mean(sigma, axis=0)

                # calculate var loss before exp
                var_loss = var_loss + tf.reduce_mean(tf.math.pow(sigma_in - s, 2))

                s = tf.math.exp(s*10)

                # calculate gaussian
                dist = tf.math.exp(-1*tf.reduce_sum(
                    tf.math.pow(spatial_emb - center, 1) * s, 1, keepdims=True))

                # print("TF", tf.reduce_sum(center))

                # apply lovasz-hinge loss
                instance_loss = instance_loss + lovasz_hinge_x(dist*2-1, in_mask[..., tf.newaxis]) # TODO: have to do this

                # seed loss
                seed_loss += self.foreground_weight * tf.reduce_sum(
                    tf.math.pow(seed_map[in_mask] - dist[in_mask], 2))

                obj_count += 1



            if obj_count > 0:
                instance_loss /= obj_count
                var_loss /= obj_count

            seed_loss = seed_loss / tf.cast(num_vertices, tf.float32)

            # print(instance_loss, var_loss, seed_loss)


            loss += w_inst * instance_loss + w_var * var_loss + w_seed * seed_loss
        loss = tf.reduce_sum(beta_values)*0 + loss / (b+1)

        return loss
