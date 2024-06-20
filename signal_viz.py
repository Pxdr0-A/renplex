import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np

samples = 512

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
  point = 0
  epochs = 32
  lr_re = "0.1999"
  lr_im = "0.0063"

  df_x = pd.read_csv(f"./out/signal_rec/x{point}_lr_{lr_re}_{lr_im}_signal_{epochs}e.csv", header=None)
  (x_re, x_im) = filter_complex_dataframe(df_x)

  df_y = pd.read_csv(f"./out/signal_rec/y{point}_lr_{lr_re}_{lr_im}_signal_{epochs}e.csv", header=None)
  (y_re, y_im) = filter_complex_dataframe(df_y)

  df_yp = pd.read_csv(f"./out/signal_rec/yp{point}_lr_{lr_re}_{lr_im}_signal_{epochs}e.csv", header=None)
  (yp_re, yp_im) = filter_complex_dataframe(df_yp)

  line = 2.5

  plt.figure(figsize=(6.5, 6.5))

  plt.plot(x_re, x_im, '.', linewidth=1)
  plt.plot(yp_re, yp_im, 'y', linewidth=line)
  plt.plot(y_re, y_im, 'g', linewidth=line)

  plt.show()

if __name__ == "__main__":
  main()