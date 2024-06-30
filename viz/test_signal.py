import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

def filter_complex_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
  samples = 512
  col = df.columns[0]
  
  df_re = pd.DataFrame()
  df_im = pd.DataFrame()

  df0 = df[col].apply(lambda x: ast.literal_eval(x.replace(" ", ",")))
  df_re[col] = df0.apply(lambda x: x[0])
  df_im[col] = df0.apply(lambda x: x[1])
  # real -> x[0], imaginary -> x[1], norm -> np.linalg.norm(x), phase -> np.arctan2(x[1], x[0])
  
  return (
    df_re.values, df_im.values
  )

def main():

  x = np.linspace(0.00, 0.05)
  plt.figure()
  plt.plot(x, (9.9*x)**4, '.')
  plt.plot(x, x, '.')
  plt.show()  

if __name__ == "__main__":
  main()