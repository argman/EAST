import tensorflow as tf

# required for both training and testing
tf.app.flags.DEFINE_string('checkpoint_path', 'models/east_icdar2015_resnet_v1_50_rbox',
                           'model folder that contains checkpoint, index and meta')
tf.app.flags.DEFINE_integer('text_scale', 512, '')

# testing parameters
tf.app.flags.DEFINE_string('test_data_path', 'training_samples',
                           'folder that contains images to test')
tf.app.flags.DEFINE_string('output_dir', 'outputs',
                           'result will be written to this folder')
tf.app.flags.DEFINE_bool('no_write_images', False,
                         'do not write images')

# training parameters
tf.app.flags.DEFINE_boolean('restore', False,
                            'whether to restore from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 14, '')
tf.app.flags.DEFINE_integer('num_readers', 16, '')
tf.app.flags.DEFINE_string('training_data_path', './training_samples/',
                           'training dataset to use')
tf.app.flags.DEFINE_string('pretrained_model_path', './models/resnet_v1_50.ckpt', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_integer('max_image_large_side', 1280, 'max image size of training')
tf.app.flags.DEFINE_integer('max_text_size', 800,
                            'if the text in the input image is bigger than this, '
                            'then we resize the image according to this')
tf.app.flags.DEFINE_integer('min_text_size', 10,
                            'if the text size is smaller than this, we ignore it during training')
tf.app.flags.DEFINE_float('min_crop_side_ratio', 0.1,
                          'when doing random crop from input image, '
                          'the min length of min(H, W)')
tf.app.flags.DEFINE_string('geometry', 'RBOX',
                           'which geometry to generate, RBOX or QUAD')
