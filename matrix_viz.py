import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
  df_train_loss = pd.read_csv("out/lc_train_loss.csv", header=None, names=["vals"])
  df_train_acc = pd.read_csv("out/lc_train_acc.csv", header=None, names=["vals"])
  df_test_loss = pd.read_csv("out/lc_test_loss.csv", header=None, names=["vals"])
  df_test_acc = pd.read_csv("out/lc_test_acc.csv", header=None, names=["vals"])

  plt.figure()
  plt.plot(df_train_loss["vals"], '.')
  plt.plot(df_test_loss["vals"], '.')

  plt.figure()
  plt.plot(df_train_acc["vals"], '.')
  plt.plot(df_test_acc["vals"], '.')

  plt.show()

if __name__ == "__main__":
  main()