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
  name = "./out/conv_tests/conv_csobel.csv"
  df_image = pd.read_csv(name, header=None)
  df_image.drop(26, axis=1, inplace=True)
  image = filter_complex_dataframe(df_image)

  plt.figure()
  plt.imshow(image)
  plt.axis('off')
  
  plt.show()

if __name__ == "__main__":
  main()