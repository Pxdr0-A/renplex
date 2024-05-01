import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
  df_image = pd.read_csv("./out/original_d.csv")
  df_image_pool = pd.read_csv("./out/dconv_image.csv")

  plt.figure()
  plt.imshow(df_image.values, cmap='hot', interpolation='nearest')
  plt.figure()
  plt.imshow(df_image_pool.values, cmap='hot', interpolation='nearest')
  
  plt.show()

if __name__ == "__main__":
  main()