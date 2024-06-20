import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

def main():
  xr_train = pd.read_csv("sig_rec/xr_train.csv", index_col=0)
  yr_train = pd.read_csv("sig_rec/yr_train.csv", index_col=0)
  xi_train = pd.read_csv("sig_rec/xi_train.csv", index_col=0)
  yi_train = pd.read_csv("sig_rec/yi_train.csv", index_col=0)
  xr_test = pd.read_csv("sig_rec/xr_test.csv", index_col=0)
  yr_test = pd.read_csv("sig_rec/yr_test.csv", index_col=0)
  xi_test = pd.read_csv("sig_rec/xi_test.csv", index_col=0)
  yi_test = pd.read_csv("sig_rec/yi_test.csv", index_col=0)

  point = 12
  samples = 512

  narrows = range(0, samples, 63)
  rem = 7
  plt.figure(figsize=(6.5, 6.5))
  # x
  plt.plot(xr_train.values[point], xi_train.values[point], '.')
  # y
  plt.plot(yr_train.values[point], yi_train.values[point])
  for i in narrows:
    plt.annotate(
      '', 
      xy=(yr_train.values[point][i], yi_train.values[point][i]), 
      xytext=(yr_train.values[point][i+rem], yi_train.values[point][i+rem]),
      arrowprops=dict(arrowstyle="->", lw=2.5, color="orange"),
      fontsize=24
    )
  

  plt.figure(figsize=(6.5, 6.5))
  # x
  plt.plot(xr_test.values[point], xi_test.values[point], '.')
  # y
  plt.plot(yr_test.values[point], yi_test.values[point])
  for i in narrows:
    plt.annotate(
      '', 
      xy=(yr_test.values[point][i], yi_test.values[point][i]), 
      xytext=(yr_test.values[point][i+rem], yi_test.values[point][i+rem]),
      arrowprops=dict(arrowstyle="->", lw=2.5, color="orange"),
      fontsize=24
    )

  plt.show()

if __name__ == "__main__":
  main()