import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def lr_studies_dense():
  lr_list = [
    #"0.1", "0.5",
    #"1", "1.4", "1.7", 
    #"2", "2.5", "2.7",
    #"3", "3.3", "3.7",
    # Start of Optimal Regime
    "4", "4.3", "4.7",
    "5", "5.3", "5.7",
    "6", "6.3", "6.7",
    "7", "7.3",
    # A bit caotic but still good results 
    "7.7", "8", "8.5",
    # End of Optimal Regime
    #"9", "9.5", "10"
  ]
  
  plt.figure()
  for lr in lr_list:
    #df_train_loss = pd.read_csv(f"out/loss_{lr}_0.csv", header=None, names=["vals"])
    #df_train_acc = pd.read_csv(f"out/acc_{lr}_0.csv", header=None, names=["vals"])
    #df_test_loss = pd.read_csv(f"out/test_loss_{lr}_0.csv", header=None, names=["vals"])
    df_test_acc = pd.read_csv(f"out/lr_studies_/test_acc_{lr}_0.csv", header=None, names=["vals"])

    #plt.figure()
    #plt.plot(df_train_loss["vals"], '.')
    #plt.plot(df_test_loss["vals"], '.')

    #plt.figure()
    #plt.plot(df_train_acc["vals"], '.')
    plt.plot(df_test_acc["vals"], '-o', label=lr)

  plt.legend()
  plt.show()

def main():
  pass

if __name__ == "__main__":
  main()