import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
  df = pd.read_csv("dataset.csv")
  
  plt.figure()
  for i in np.unique(df["class"]):
    plt.plot(
      df[df["class"] == i]["feature0"],
      df[df["class"] == i]["feature1"],
      "o"
    )
  
  plt.show()

if __name__ == "__main__":
  main()