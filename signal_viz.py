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
  point = 9
  epochs = 32
  seed = 891298565
  lr_re = "0.0350"
  lr_im = "0.0011"

  df_x = pd.read_csv(f"./out/signal_rec/{seed}_x{point}_lr_{lr_re}_{lr_im}_signal_{epochs}e.csv", header=None)
  (xr_test, xi_test) = filter_complex_dataframe(df_x)

  df_y = pd.read_csv(f"./out/signal_rec/{seed}_y{point}_lr_{lr_re}_{lr_im}_signal_{epochs}e.csv", header=None)
  (yr_test, yi_test) = filter_complex_dataframe(df_y)

  df_yp = pd.read_csv(f"./out/signal_rec/{seed}_yp{point}_lr_{lr_re}_{lr_im}_signal_{epochs}e.csv", header=None)
  (re_pred, im_pred) = filter_complex_dataframe(df_yp)

  narrows = range(0, samples, 63)
  rem = 7
  lw = 2.5
  lim = 1.1
  ticks = np.round(np.linspace(-lim, lim, 8), 1)

  plt.figure(figsize=(6.5, 6.5))
  plt.xticks(ticks)
  plt.yticks(ticks)
  plt.xlim((-lim, lim))
  plt.ylim((-lim, lim))
  plt.xlabel(r"$\Re\left\{ s(t) \right\}$", fontsize=14)
  plt.ylabel(r"$\Im\left\{ s(t) \right\}$", fontsize=14)
  plt.tick_params(axis='both', which='major', labelsize=12)
  plt.plot(xr_test, xi_test, '.b', label=r"$x(t)$", alpha=0.6)
  plt.plot(re_pred, im_pred, '-y', linewidth=lw, label=r"$s(t)$")
  for j in narrows:
    plt.annotate(
      '', 
      xy=(re_pred[j], im_pred[j]), 
      xytext=(re_pred[j+rem], im_pred[j+rem]),
      arrowprops=dict(arrowstyle="->", lw=2.5, color="yellow"),
      fontsize=24
    )
  plt.plot(yr_test, yi_test, '--g', linewidth=lw, label=r"$y(t)$")
  for j in narrows:
    plt.annotate(
      '', 
      xy=(yr_test[j], yi_test[j]), 
      xytext=(yr_test[j+rem], yi_test[j+rem]),
      arrowprops=dict(arrowstyle="->", lw=2.5, color="green"),
      fontsize=24
    )
  plt.legend(fontsize=14)
  plt.show()


  plt.show()

if __name__ == "__main__":
  main()