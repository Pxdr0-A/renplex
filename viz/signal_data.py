import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

def main():
  xr_train = pd.read_csv("sig_rec/xr_train.csv", index_col=0)
  yr_train = pd.read_csv("sig_rec/yr_train.csv", index_col=0)
  xi_train = pd.read_csv("sig_rec/xi_train.csv", index_col=0)
  yi_train = pd.read_csv("sig_rec/yi_train.csv", index_col=0)
  #xr_test = pd.read_csv("sig_rec/xr_test.csv", index_col=0)
  #yr_test = pd.read_csv("sig_rec/yr_test.csv", index_col=0)
  #xi_test = pd.read_csv("sig_rec/xi_test.csv", index_col=0)
  #yi_test = pd.read_csv("sig_rec/yi_test.csv", index_col=0)

  point = 15
  samples = 512

  narrows = range(0, samples, 63)
  rem = 7
  lim = 1.1
  lw = 2.5
  ticks = np.round(np.linspace(-lim, lim, 8), 1)

  plt.figure(figsize=(6.5, 6.5))
  plt.xticks(ticks)
  plt.yticks(ticks)
  plt.xlim((-lim, lim))
  plt.ylim((-lim, lim))
  plt.xlabel(r"$\Re\left\{ y(t) \right\}$", fontsize=14)
  plt.ylabel(r"$\Im\left\{ y(t) \right\}$", fontsize=14)
  plt.tick_params(axis='both', which='major', labelsize=12)

  # x
  plt.plot(xr_train.values[point], xi_train.values[point], '.b', label=r"$x(t)$", alpha=0.6)
  # y
  plt.plot(yr_train.values[point], yi_train.values[point], '--g', linewidth=lw, label=r"$y(t)$")
  for i in narrows:
    plt.annotate(
      '', 
      xy=(yr_train.values[point][i], yi_train.values[point][i]), 
      xytext=(yr_train.values[point][i+rem], yi_train.values[point][i+rem]),
      arrowprops=dict(arrowstyle="->", lw=2.5, color="green"),
      fontsize=24
    )
  
  plt.legend(fontsize=14)
  plt.show()

if __name__ == "__main__":
  main()