import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
  df_image = pd.read_csv("./out/conv_tests/original.csv")
  df_image_pool = pd.read_csv("./out/conv_tests/max_pool.csv")
  df_image_up = pd.read_csv("./out/conv_tests/max_pool_upsampled.csv")

  plt.figure()
  plt.imshow(df_image.values, cmap='hot', interpolation='nearest')
  plt.figure()
  plt.imshow(df_image_pool.values, cmap='hot', interpolation='nearest')
  plt.figure()
  plt.imshow(df_image_up.values, cmap='hot', interpolation='nearest')
  
  plt.show()

if __name__ == "__main__":
  main()