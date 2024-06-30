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
    435726692,
    328557473,
    348769349,
    224783561,
    981347827
  ]

  samples = 512
  input_units = 128
  inter_encoding = 32
  min_encoding = 16
  model_id = 3

  # for viz  
  narrows = range(0, samples, 63)
  rem = 7
  lw = 2.5
  lim = 1.1
  ticks = np.round(np.linspace(-lim, lim, 8), 1)

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
      tf.keras.layers.Dense(input_units, activation="tanh", dtype='float32'),
      tf.keras.layers.Dense(inter_encoding, activation="tanh", dtype='float32'),
      tf.keras.layers.Dense(min_encoding, activation="tanh", dtype='float32'),
      tf.keras.layers.Dense(inter_encoding, activation="tanh", dtype='float32'),
      tf.keras.layers.Dense(samples, activation=None, dtype='float32')
    ])

    modeli = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(input_units, activation="tanh", dtype='float32'),
      tf.keras.layers.Dense(inter_encoding, activation="tanh", dtype='float32'),
      tf.keras.layers.Dense(min_encoding, activation="tanh", dtype='float32'),
      tf.keras.layers.Dense(inter_encoding, activation="tanh", dtype='float32'),
      tf.keras.layers.Dense(samples, activation=None, dtype='float32')
    ])

    loss_fn = 'mean_squared_error'
    lr = 17.5
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(
      optimizer=optimizer,
      loss=loss_fn
    )

    loss_fn = 'mean_squared_error'
    lr = 17.5
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    modeli.compile(
      optimizer=optimizer,
      loss=loss_fn
    )

    loss_vals = []
    # train
    epochs = 32
    for j in range(epochs):
      print("Epoch: ", j)
      print("Seed:", seed)

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

      re_pred = model.predict(xr_train)
      im_pred = modeli.predict(xi_train)

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
    
    for i in range(10):
      plt.figure(figsize=(6.5, 6.5))
      plt.xticks(ticks)
      plt.yticks(ticks)
      plt.xlim((-lim, lim))
      plt.ylim((-lim, lim))
      plt.xlabel(r"$\Re\left\{ s(t) \right\}$", fontsize=14)
      plt.ylabel(r"$\Im\left\{ s(t) \right\}$", fontsize=14)
      plt.tick_params(axis='both', which='major', labelsize=12)
      plt.plot(xr_train.values[i], xi_train.values[i], '.b', label=r"$x(t)$", alpha=0.6)
      plt.plot(re_pred[i], im_pred[i], '-y', linewidth=lw, label=r"$s(t)$")
      for j in narrows:
        plt.annotate(
          '', 
          xy=(re_pred[i][j], im_pred[i][j]), 
          xytext=(re_pred[i][j+rem], im_pred[i][j+rem]),
          arrowprops=dict(arrowstyle="->", lw=2.5, color="yellow"),
          fontsize=24
        )
      plt.plot(yr_train.values[i], yi_train.values[i], '--g', linewidth=lw, label=r"$y(t)$")
      for j in narrows:
        plt.annotate(
          '', 
          xy=(yr_train.values[i][j], yi_train.values[i][j]), 
          xytext=(yr_train.values[i][j+rem], yi_train.values[i][j+rem]),
          arrowprops=dict(arrowstyle="->", lw=2.5, color="green"),
          fontsize=24
        )
      plt.legend(fontsize=14)
      plt.show()


def main():
  #save_datasets()
  run_model()

if __name__ == "__main__":
  main()