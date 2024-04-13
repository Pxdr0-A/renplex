import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
  df_image = pd.read_csv("./out/original.csv")
  df_image_conv = pd.read_csv("./out/conv_image.csv")


  plt.figure()
  plt.imshow(df_image.values, cmap='hot', interpolation='nearest')
  plt.figure()
  plt.imshow(df_image_conv.values, cmap='hot', interpolation='nearest')
  plt.show()

if __name__ == "__main__":
  main()