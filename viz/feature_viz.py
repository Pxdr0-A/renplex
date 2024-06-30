import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

def filter_complex_dataframe(df: pd.DataFrame) -> pd.DataFrame:
  cols = df.columns
  df0 = pd.DataFrame()
  for col in cols:
    df0[col] = df[col].apply(lambda x: ast.literal_eval(x.replace(" ", ",")))

  cols0 = df0.columns
  for col0 in cols0:
    df0[col0] = df0[col0].apply(lambda x: [int(255 * (x[0])), int(255 * (x[1])), int(0)]) 
    # real -> x[0], imaginary -> x[1], norm -> np.linalg.norm(x), phase -> np.arctan2(x[1], x[0])

  image = []
  for row in df0.values:
    image_row = []
    for elm in row:
      image_row.append(elm)
    
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
  # 981347827
  # 224783561
  epochs = 16
  model_id = 2
  seed = seed_list[4]

  df_image1 = pd.read_csv(f"./out/complex_features/conv{model_id}/{seed}_2_{epochs}e_original.csv", header=None)
  df_image1.drop(28, axis=1, inplace=True)
  df1 = filter_complex_dataframe(df_image1)
  
  df_image2 = pd.read_csv(f"./out/complex_features/conv{model_id}/{seed}_4_{epochs}e_original.csv", header=None)
  df_image2.drop(28, axis=1, inplace=True)
  df2 = filter_complex_dataframe(df_image2)

  plt.figure()
  plt.imshow(df1)
  plt.axis("off")

  plt.figure()
  plt.imshow(df2)
  plt.axis("off")

  for i in range(8):
    df_feature = pd.read_csv(
      f"./out/complex_features/conv{model_id}/{seed}_2_{epochs}e_feature_{i}_{1}.csv", header=None
    )
    df_feature.drop(26, axis=1, inplace=True)
    df = filter_complex_dataframe(df_feature)

    plt.figure()
    plt.axis('off')
    plt.imshow(df)

    df_feature = pd.read_csv(
      f"./out/complex_features/conv{model_id}/{seed}_4_{epochs}e_feature_{i}_{1}.csv", header=None
    )
    df_feature.drop(26, axis=1, inplace=True)
    df = filter_complex_dataframe(df_feature)

    plt.figure()
    plt.axis('off')
    plt.imshow(df)
  
  plt.show()

if __name__ == "__main__":
  main()