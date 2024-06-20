import random
import csv
import ast

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

import tensorflow as tf

def filter_complex_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  samples = 512
  cols = df.columns
  
  df_re = pd.DataFrame()
  df_im = pd.DataFrame()
  for col in cols:
    df0 = df[col].apply(lambda x: ast.literal_eval(x.replace(" ", ",")))

    df_re[col] = df0.apply(lambda x: x[0])
    df_im[col] = df0.apply(lambda x: x[1])
    # real -> x[0], imaginary -> x[1], norm -> np.linalg.norm(x), phase -> np.arctan2(x[1], x[0])
  
  return (
    df_re[cols[0:samples]], df_re[cols[(samples):(2*samples)]], 
    df_im[cols[0:samples]], df_im[cols[(samples):(2*samples)]]
  )

def save_datasets():
  # get dataset
  train = pd.read_csv("train.csv", header=None)
  (xr_train, yr_train, xi_train, yi_train) = filter_complex_dataframe(train)
  xr_train.to_csv("sig_rec/xr_train.csv")
  yr_train.to_csv("sig_rec/yr_train.csv")
  xi_train.to_csv("sig_rec/xi_train.csv")
  yi_train.to_csv("sig_rec/yi_train.csv")

  test = pd.read_csv("test.csv", header=None)
  (xr_test, yr_test, xi_test, yi_test) = filter_complex_dataframe(test)
  xr_test.to_csv("sig_rec/xr_test.csv")
  yr_test.to_csv("sig_rec/yr_test.csv")
  xi_test.to_csv("sig_rec/xi_test.csv")
  yi_test.to_csv("sig_rec/yi_test.csv")

def run_model():
  seed_list = [
    891298565,
    #435726692,
    #328557473,
    #348769349,
    #224783561,
    #981347827
  ]

  samples = 512
  inter_encoding = 16
  min_encoding = 8
  model_id = 3

  xr_train = pd.read_csv("sig_rec/xr_train.csv", index_col=0)
  yr_train = pd.read_csv("sig_rec/yr_train.csv", index_col=0)
  xi_train = pd.read_csv("sig_rec/xi_train.csv", index_col=0)
  yi_train = pd.read_csv("sig_rec/yi_train.csv", index_col=0)
  xr_test = pd.read_csv("sig_rec/xr_test.csv", index_col=0)
  yr_test = pd.read_csv("sig_rec/yr_test.csv", index_col=0)
  xi_test = pd.read_csv("sig_rec/xi_test.csv", index_col=0)
  yi_test = pd.read_csv("sig_rec/yi_test.csv", index_col=0)

  for seed in seed_list:
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # define model
    model = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(samples, activation="tanh", dtype='float32'),
      tf.keras.layers.Dense(inter_encoding, activation="tanh", dtype='float32'),
      tf.keras.layers.Dense(min_encoding, activation="tanh", dtype='float32'),
      tf.keras.layers.Dense(inter_encoding, activation="tanh", dtype='float32'),
      tf.keras.layers.Dense(samples, activation="tanh", dtype='float32')
    ])

    modeli = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(samples, activation="tanh", dtype='float32'),
      tf.keras.layers.Dense(inter_encoding, activation="tanh", dtype='float32'),
      tf.keras.layers.Dense(min_encoding, activation="tanh", dtype='float32'),
      tf.keras.layers.Dense(inter_encoding, activation="tanh", dtype='float32'),
      tf.keras.layers.Dense(samples, activation="tanh", dtype='float32')
    ])

    loss_fn = 'mean_squared_error'
    lr = 3.0
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(
      optimizer=optimizer,
      loss=loss_fn
    )

    loss_fn = 'mean_squared_error'
    lr = 3.0
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    modeli.compile(
      optimizer=optimizer,
      loss=loss_fn
    )

    loss_vals = []
    # train
    epochs = 16
    for i in range(epochs):
      model.fit(
        x=xr_train,
        y=yr_train,
        batch_size=100,
        epochs=1
      )
      modeli.fit(
        x=xi_train,
        y=yi_train,
        batch_size=100,
        epochs=1
      )

      re_pred = model.predict(xr_test)
      im_pred = modeli.predict(xi_test)

      pred = re_pred + 1j*im_pred
      clean_signal = yr_test + 1j*yi_test
      
      loss = np.mean(np.abs(pred - clean_signal)**2)
      loss_vals.append(loss)
      print("Total loss: ", loss)
    
    with open(f'tensorflow/{seed}_id{model_id}_tloss_{lr}_{epochs}e.csv', 'w', newline='') as csvfile:
      fieldnames = ['vals', 'empty']
      csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
      csv_writer.writeheader()
      for val in loss_vals:
        csv_writer.writerow({'vals': val, 'empty': ''})
    
    plt.figure()
    plt.plot(xr_test.values[1], xi_test.values[1], '.')
    plt.plot(yr_test.values[1], yi_test.values[1])
    plt.plot(re_pred[1], im_pred[1])
    plt.show()


def main():
  save_datasets()
  #run_model()

if __name__ == "__main__":
  main()