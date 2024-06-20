import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

def filter_complex_dataframe(df: pd.DataFrame) -> pd.DataFrame:
  cols = df.columns
  df0 = pd.DataFrame()
  for index in range(0, len(cols)-1, 2):
    df0[str(index)] = [[0, 0]] + [ast.literal_eval(s) for s in df[cols[index]] + "," + df[cols[index+1]]]
  
  cols0 = df0.columns
  for col0 in cols0:
    df0[col0] = df0[col0].apply(lambda x: [int(255 * (np.linalg.norm(x))), int(255 * (np.arctan2(x[1], x[0]))), int(0)]) 
    # real -> x[0], imaginary -> x[1], norm -> np.linalg.norm(x), phase -> np.arctan2(x[1], x[0])

  image = []
  for row in df0.values:
    image_row = []
    for rgb in row:
      pixel = []
      for channel in rgb:
        pixel.append(int(channel))
      
      image_row.append(pixel)

    image.append(image_row)
  
  return image


def main():
  seed_list = [
    891298565,
    435726692,
    328557473,
    348769349,
    224783561,
    981347827
  ]

  epochs = 1
  model_id = 2

  df_image1 = pd.read_csv(f"./out/complex_features/conv{model_id}/{seed_list[0]}_2_{epochs}e_original.csv")
  df1 = filter_complex_dataframe(df_image1)

  df_image2 = pd.read_csv(f"./out/complex_features/conv{model_id}/{seed_list[0]}_4_{epochs}e_original.csv")
  df2 = filter_complex_dataframe(df_image2)

  plt.figure()
  plt.imshow(df1)

  plt.figure()
  plt.imshow(df2)

  for i in range(8):
    df_feature = pd.read_csv(
      f"./out/complex_features/conv{model_id}/{seed_list[0]}_2_{epochs}e_feature_{i}_{3}.csv"
    )
    df = filter_complex_dataframe(df_feature)

    plt.figure()
    plt.imshow(df)

    df_feature = pd.read_csv(
      f"./out/complex_features/conv{model_id}/{seed_list[0]}_4_{epochs}e_feature_{i}_{3}.csv"
    )
    df = filter_complex_dataframe(df_feature)

    plt.figure()
    plt.imshow(df)
  
  plt.show()

if __name__ == "__main__":
  main()