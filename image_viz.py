import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
  df_image = pd.read_csv("./out/original.csv")
  df_image_conv = pd.read_csv("./out/conv_image.csv")
  df_image_rev = pd.read_csv("./out/conv_image_rev.csv")
  df_image_rev1 = pd.read_csv("./out/conv_image_rev1.csv")
  df_image_conv1 = pd.read_csv("./out/conv_image1.csv")

  plt.figure()
  plt.imshow(df_image.values, cmap='hot', interpolation='nearest')
  plt.figure()
  plt.imshow(df_image_conv.values, cmap='hot', interpolation='nearest')
  plt.figure()
  plt.imshow(df_image_rev.values, cmap='hot', interpolation='nearest')
  plt.figure()
  plt.imshow(df_image_rev1.values, cmap='hot', interpolation='nearest')
  plt.figure()
  plt.imshow(df_image_conv1.values, cmap='hot', interpolation='nearest')
  
  plt.show()

if __name__ == "__main__":
  main()