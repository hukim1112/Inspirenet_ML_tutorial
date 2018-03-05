"""Example code for TensorFlow Wide & Deep Tutorial using tf.estimator API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys

import tensorflow as tf

_CSV_COLUMNS = [
    'UserID', 'MovieID', 'Rating', 'Timestamp', 'Gender', 'Age', 'Occupation',
    'Zipcode', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror', 'Musical', 'Mystery',
    'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western'
]


_CSV_COLUMN_DEFAULTS = [[''], [''], [0], [0], [''], [''], [''],
                        [''], [''], [''], [''], [''], [''], [''],
                        [''], [''], [''], [''], [''], [''], [''],
                        [''], [''], [''], [''], ['']]

_CSV_COLUMNS_PD = [
    'UserID', 'MovieID', 'Timestamp', 'Gender', 'Age', 'Occupation',
    'Zipcode', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror', 'Musical', 'Mystery',
    'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western'
]


_CSV_COLUMN_DEFAULTS_PD = [[''], [''], [0], [''], [''], [''],
                          [''], [''], [''], [''], [''], [''], [''],
                          [''], [''], [''], [''], [''], [''], [''],
                          [''], [''], [''], [''], ['']]


parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='/home/dongsu/Tensorflow/movie_lens/wide_deep_model',
    help='Base directory for the model.')

parser.add_argument(
    '--model_type', type=str, default='wide_deep',
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument(
    '--train_epochs', type=int, default=40, help='Number of training epochs.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=2,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=1000, help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='/home/dongsu/Tensorflow/movie_lens/train_df',
    help='Path to the training data.')

parser.add_argument(
    '--test_data', type=str, default='/home/dongsu/Tensorflow/movie_lens/test_df',
    help='Path to the test data.')

parser.add_argument(
    '--prediction_data', type=str, default='/home/dongsu/Tensorflow/movie_lens/pred_df_tf',
    help='Path to the prediction data.')

_NUM_EXAMPLES = {
    'train': 750149,
    #'train': 75014,
    #'validation': 375075,
}


def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  # Continuous columns
  Timestamp = tf.feature_column.numeric_column('Timestamp')

  Gender = tf.feature_column.categorical_column_with_vocabulary_list(
      'Gender', ['M', 'F'])

  Age = tf.feature_column.categorical_column_with_vocabulary_list(
      'Age', ['1', '18', '25', '35', '45', '50', '56'])

  Occupation = tf.feature_column.categorical_column_with_hash_bucket(
      'Occupation', hash_bucket_size=1000)

  Zipcode = tf.feature_column.categorical_column_with_hash_bucket(
      'Zipcode', hash_bucket_size=10000)

  MovieID = tf.feature_column.categorical_column_with_hash_bucket(
      'MovieID', hash_bucket_size=10000)

  UserID = tf.feature_column.categorical_column_with_hash_bucket(
      'UserID', hash_bucket_size=10000)

  Action = tf.feature_column.categorical_column_with_vocabulary_list(
      'Action', ['0', '1'])

  Adventure = tf.feature_column.categorical_column_with_vocabulary_list(
      'Adventure', ['0', '1'])

  Animation = tf.feature_column.categorical_column_with_vocabulary_list(
      'Animation', ['0', '1'])

  Children = tf.feature_column.categorical_column_with_vocabulary_list(
      'Children', ['0', '1'])

  Comedy = tf.feature_column.categorical_column_with_vocabulary_list(
      'Comedy', ['0', '1'])

  Crime = tf.feature_column.categorical_column_with_vocabulary_list(
      'Crime', ['0', '1'])

  Documentary = tf.feature_column.categorical_column_with_vocabulary_list(
      'Documentary', ['0', '1'])

  Drama = tf.feature_column.categorical_column_with_vocabulary_list(
      'Drama', ['0', '1'])

  Fantasy = tf.feature_column.categorical_column_with_vocabulary_list(
      'Fantasy', ['0', '1'])

  Film_Noir = tf.feature_column.categorical_column_with_vocabulary_list(
      'Film_Noir', ['0', '1'])

  Horror = tf.feature_column.categorical_column_with_vocabulary_list(
      'Horror', ['0', '1'])

  Musical = tf.feature_column.categorical_column_with_vocabulary_list(
      'Musical', ['0', '1'])

  Mystery = tf.feature_column.categorical_column_with_vocabulary_list(
      'Mystery', ['0', '1'])

  Romance = tf.feature_column.categorical_column_with_vocabulary_list(
      'Romance', ['0', '1'])

  Sci_Fi = tf.feature_column.categorical_column_with_vocabulary_list(
      'Sci_Fi', ['0', '1'])

  Thriller = tf.feature_column.categorical_column_with_vocabulary_list(
      'Thriller', ['0', '1'])

  War = tf.feature_column.categorical_column_with_vocabulary_list(
      'War', ['0', '1'])

  Western = tf.feature_column.categorical_column_with_vocabulary_list(
       'Western', ['0', '1'])

  crossed_column_genres = tf.feature_column.crossed_column(
      ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
      'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror', 'Musical',
      'Mystery', 'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western'], hash_bucket_size=10000)

  zipcode_genres = tf.feature_column.crossed_column([crossed_column_genres, 'Zipcode'], hash_bucket_size=10000)

  userid_zipcode = tf.feature_column.crossed_column(['UserID', 'Zipcode'], hash_bucket_size=10000)


  # Wide columns and deep columns.
  base_columns = [
        UserID, Zipcode, userid_zipcode, zipcode_genres



    # Wide columns and deep columns.
  ]

  crossed_columns = [
      tf.feature_column.crossed_column(
          [crossed_column_genres, 'UserID'], hash_bucket_size=100000),
      tf.feature_column.crossed_column(
          ['UserID', zipcode_genres], hash_bucket_size=10000),
      tf.feature_column.crossed_column(
          [userid_zipcode, zipcode_genres], hash_bucket_size=10000),

          ]

  wide_columns = base_columns + crossed_columns

  deep_columns = [
      #Timestamp,
      #tf.feature_column.indicator_column(Age),
      #tf.feature_column.indicator_column(Gender),
      #tf.feature_column.indicator_column(Occupation),
      tf.feature_column.embedding_column(Zipcode, dimension=8),
      tf.feature_column.embedding_column(UserID, dimension=8),
      #tf.feature_column.embedding_column(Action, dimension=4),
      #tf.feature_column.embedding_column(Animation, dimension=4),
      #tf.feature_column.embedding_column(Adventure, dimension=4),
      #tf.feature_column.embedding_column(Children, dimension=4),
      #tf.feature_column.embedding_column(Comedy, dimension=4),
      #tf.feature_column.embedding_column(Crime, dimension=4),
      #tf.feature_column.embedding_column(Documentary, dimension=4),
      #tf.feature_column.embedding_column(Drama, dimension=4),
      #tf.feature_column.embedding_column(Fantasy, dimension=4),
      #tf.feature_column.embedding_column(Film_Noir, dimension=4),
      #tf.feature_column.embedding_column(Horror, dimension=4),
      #tf.feature_column.embedding_column(Musical, dimension=4),
      #tf.feature_column.embedding_column(Mystery, dimension=4),
      #tf.feature_column.embedding_column(Romance, dimension=4),
      #tf.feature_column.embedding_column(Sci_Fi, dimension=4),
      #tf.feature_column.embedding_column(Thriller, dimension=4),
      #tf.feature_column.embedding_column(War, dimension=4),
      #tf.feature_column.embedding_column(Western, dimension=4),
      #tf.feature_column.embedding_column(UserID, dimension=8),
  ]
  return wide_columns, deep_columns


def build_estimator(model_dir, model_type):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = build_model_columns()

  hidden_units = [100, 50]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

  if model_type == 'wide':
    return tf.estimator.LinearRegressor(
        model_dir=model_dir,
        feature_columns=wide_columns,
        #n_classes=5,
        config=run_config)
  elif model_type == 'deep':
    return tf.estimator.DNNRegressor(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        #n_classes=5,
        config=run_config)
  else:
    return tf.estimator.DNNLinearCombinedRegressor(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        #n_classes=5,
        config=run_config)


def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have either run data_download.py or '
      'set both arguments --train_data and --test_data.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    target = features.pop('Rating')
    return features, target

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)


  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)


  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  return dataset

def pred_input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have either run data_download.py or '
      'set both arguments --train_data and --test_data.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS_PD)
    features = dict(zip(_CSV_COLUMNS_PD, columns))
    return features

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  return dataset


def main(unused_argv):
  # Clean up the model directory if present
  shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
  model = build_estimator(FLAGS.model_dir, FLAGS.model_type)

  # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
  # FLAGS.train_epochs // FLAGS.epochs_per_eval
  for n in range(200):
    model.train(input_fn=lambda: input_fn(
        FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))


    results = model.evaluate(input_fn=lambda: input_fn(
        FLAGS.test_data, 1, False, FLAGS.batch_size))

    prediction = model.predict(input_fn=lambda: pred_input_fn(
        FLAGS.prediction_data, 1, False, FLAGS.batch_size))

    # Display evaluation metrics
    print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
    print('-' * 60)

    for key in sorted(results):
      print('%s: %s' % (key, results[key]))

    for pre in prediction:
        print('predition : ', pre)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
