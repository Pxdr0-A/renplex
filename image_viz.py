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
    df0[col0] = df0[col0].apply(lambda x: np.linalg.norm(x))

  return df0

def main():
  df_image = pd.read_csv("./out/complex_features/lr_conv1_1_0_20e_original.csv")
  df_feature0 = pd.read_csv("./out/complex_features/lr_conv1_1_0_20e_feature_0.csv")
  df_feature1 = pd.read_csv("./out/complex_features/lr_conv1_1_0_20e_feature_1.csv")
  df_feature2 = pd.read_csv("./out/complex_features/lr_conv1_1_0_20e_feature_2.csv")  
  df_feature3 = pd.read_csv("./out/complex_features/lr_conv1_1_0_20e_feature_3.csv")  
  df_feature4 = pd.read_csv("./out/complex_features/lr_conv1_1_0_20e_feature_4.csv")
  df_feature5 = pd.read_csv("./out/complex_features/lr_conv1_1_0_20e_feature_5.csv")  
  df_feature6 = pd.read_csv("./out/complex_features/lr_conv1_1_0_20e_feature_6.csv")
  df_feature7 = pd.read_csv("./out/complex_features/lr_conv1_1_0_20e_feature_7.csv") 
  
  df0 = filter_complex_dataframe(df_image)
  df1 = filter_complex_dataframe(df_feature0)
  df2 = filter_complex_dataframe(df_feature1)
  df3 = filter_complex_dataframe(df_feature2)
  df4 = filter_complex_dataframe(df_feature3)
  df5 = filter_complex_dataframe(df_feature4)
  df6 = filter_complex_dataframe(df_feature5)
  df7 = filter_complex_dataframe(df_feature6)
  df8 = filter_complex_dataframe(df_feature7)
  
  plt.figure()
  plt.imshow(df0.values, cmap='hot', interpolation='nearest')
  plt.figure()
  plt.imshow(df1.values, cmap='hot', interpolation='nearest')
  plt.figure()
  plt.imshow(df2.values, cmap='hot', interpolation='nearest')
  plt.figure()
  plt.imshow(df3.values, cmap='hot', interpolation='nearest')
  plt.figure()
  plt.imshow(df4.values, cmap='hot', interpolation='nearest')
  plt.figure()
  plt.imshow(df5.values, cmap='hot', interpolation='nearest')
  plt.figure()
  plt.imshow(df6.values, cmap='hot', interpolation='nearest')
  plt.figure()
  plt.imshow(df7.values, cmap='hot', interpolation='nearest')
  plt.figure()
  plt.imshow(df8.values, cmap='hot', interpolation='nearest')
  
  plt.show()

if __name__ == "__main__":
  main()