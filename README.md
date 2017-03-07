###My personal TensorFlow utils - still under development, use with caution.



<hr>

####Data preprocessing pipelines for images and text
You can easily construct data augmentation pipelines and batch generators.
For example, whole ImageNet data augmentation + batch generator pipeline has less than 100 lines of code and is able to load and process about 900 images per second on my Intel 6850k CPU.


```python
    from tfutils import *

    # image_pipeline takes an image path and returns preprocessed image as np array (RGB)
    image_pipeline = compose_ops([
        load(),
        random_crop(IMAGE_SIZE, AREA_RANGE, ASPECT_RATIO_RANGE),
        probabilistic_op(0.5, horizontal_flip()),
        normalize(IMAGE_NET_PIXEL_MEAN, IMAGE_NET_PIXEL_STD),
        random_lighting(LIGHTNING_APLHA_STD, IMG_NET_PIXEL_EIG_VAL, IMG_NET_PIXEL_EIG_VEC)
    ])

    # label_pipeline takes image path and returns a label
    label_pipeline = compose_ops([
        folder_name(),
        key_to_element(SYNSET_TO_CLASS_ID)
    ])

    # example_pipeline for each input image path will return a tuple (Image, Label)
    example_pipeline = for_each(parallelise_ops([
        image_pipeline,
        label_pipeline
    ]))


    bm = BatchManager(
        example_pipeline,
        TRAIN_IMAGE_PATHS,
        generic_batch_composer(np.float32, np.int32),
        BATCH_SIZE,
        shuffle_examples=True,
        num_workers=6
    )
```

'bm' is the batch generator.

<hr>

####Easy training util

When you have your batch generators ready you can easily start the training/validation.
All you need is the TensorFlow training operation and a simple configuration.

```python
    saver = tf.train.Saver(tf.global_variables())

    nt = NiceTrainer(sess,               # TensorFlow session
                     train_bm,           # BatchManager for training data
                     [images, labels],   # outputs of the batch manager
                     train_op,           # TensorFlow training op that minimizes the loss
                     bm_val=val_bm,      # optional BatchManager for validation data
                     extra_variables={'loss': loss,
                                      'probs': probs},   # extra variables that will be calculated with each train iteration
                     printable_vars=['loss', 'top-1-acc', 'top-5-acc'],   # extra variables that should be smoothed and printed every iteration
                     computed_variables={'top-5-acc': accuracy_calc_op(n=5),
                                         'top-1-acc': accuracy_calc_op(n=1)},   # some extra values that you want to calculte using custom function

                     save_every=600,     # will perform periodic model saves every 600 seconds
                     saver=saver,        # optional model saver
                     smooth_coef=0.99
    )
    nt.restore()

    for epoch in xrange(100):
        nt.train()
        nt.validate()
        nt.save()  # <- this is not needed because model will be saved periodically anyway, but you can keep extra checks every epoch if you want
```

During training you see something like this, updated every iteration:

```
    3585/10010 - time: 0.827 - data: 0.005 - ETA: 5343 - loss: 1.5937 - top-1-acc: 0.6205 - top-5-acc: 0.8368
```