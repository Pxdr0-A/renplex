import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def lr_studies_dense():
  lr_list = [
    "1.00_0.00",
    "2.00_0.00",
    "3.00_0.00",
    "4.00_0.00",
  ]
  
  f1 = plt.figure()
  f2 = plt.figure()
  ax1 = f1.add_subplot(111)
  ax2 = f2.add_subplot(111)
  for lr in lr_list:
    #df_train_loss = pd.read_csv(f"out/loss_{lr}_0.csv", header=None, names=["vals"])
    #df_train_acc = pd.read_csv(f"out/acc_{lr}_0.csv", header=None, names=["vals"])
    df_test_loss = pd.read_csv(f"out/0/test_loss_{lr}_20e.csv", header=None, names=["vals", "end"])
    df_test_accu = pd.read_csv(f"out/0/test_acc_{lr}_20e.csv", header=None, names=["vals", "end"])
    
    #plt.plot(df_train_loss["vals"], '.')
    ax1.plot(df_test_loss["vals"], '-o', label=lr)

    #plt.plot(df_train_acc["vals"], '.')
    ax2.plot(df_test_accu["vals"], '-o', label=lr)

  ax1.legend()
  ax2.legend()
  plt.show()

def main():
  lr_studies_dense()

if __name__ == "__main__":
  main()