import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
  df = pd.read_csv("matrix.csv", header=None, names=["vals"])

  plt.figure()
  plt.plot(df["vals"], '.')
  plt.show()

if __name__ == "__main__":
  main()