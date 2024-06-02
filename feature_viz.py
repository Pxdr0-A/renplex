import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

def filter_complex_dataframe(df: pd.DataFrame) -> pd.DataFrame:
  cols = df.columns
  df0 = pd.DataFrame()
  for index in range(0, len(cols)-1, 2):
    df0[str(index)] = [ast.literal_eval(s) for s in df[cols[index]] + "," + df[cols[index+1]]]

  cols0 = df0.columns
  for col0 in cols0:
    df0[col0] = df0[col0].apply(lambda x: np.arctan2(x[1], x[0])) 
    # real -> x[0], imaginary -> x[1], norm -> np.linalg.norm(x), phase -> np.arctan2(x[1], x[0])

  return df0

def main():
  df_image = pd.read_csv("./out/complex_features/conv2/lr_1.5_0_20e_original.csv")
  df0 = filter_complex_dataframe(df_image)
  plt.figure()
  plt.imshow(df0.values, cmap='hot', interpolation='nearest')

  for i in range(8):
    df_feature = pd.read_csv(f"./out/complex_features/conv2/lr_1.5_0_20e_feature_{i}_{1}.csv")
    df = filter_complex_dataframe(df_feature)

    plt.figure()
    plt.imshow(df.values, cmap='hot', interpolation='nearest')
  
  plt.show()

if __name__ == "__main__":
  main()