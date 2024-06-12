import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def lr_studies_dense():
  lr_list = [
    "2.00_0.00",
    "2.00_0.06",
    "2.00_0.13",
    "1.98_0.31",
    "1.90_0.62",
  ]

  def format_label(x: str):
    x_new = x.replace("_", " + i")

    return fr"${x_new}$"
  
  f1 = plt.figure(figsize=(6.5, 6.5))
  f2 = plt.figure(figsize=(6.5, 6.5))
  ax1 = f1.add_subplot(111)
  ax2 = f2.add_subplot(111)
  
  # loss and accuracy zoom region
  e1, e2 = 12, 22
  l1, l2 = 0.004, 0.010
  a1, a2 = 0.95, 0.975

  ax1_inset = inset_axes(
    ax1,
    width="33%", height="33%",
    # (left, bottom, width, height)
    bbox_to_anchor=(-0.05, -0.45, 1.0, 1.0), bbox_transform=ax1.transAxes
  )
  ax1_inset.set_xlim(e1, e2)
  ax1_inset.set_ylim(l1, l2)
  ax1_inset.tick_params(axis='both', which='major', labelsize=12)


  ax2_inset = inset_axes(
    ax2,
    width="33%", height="33%",
    # (left, bottom, width, height)
    bbox_to_anchor=(-0.05, -0.20, 1.0, 1.0), bbox_transform=ax2.transAxes
  )
  ax2_inset.set_xlim(e1, e2)
  ax2_inset.set_ylim(a1, a2)
  ax2_inset.tick_params(axis='both', which='major', labelsize=12)

  e = list(range(1,20+1))
  for lr in lr_list:
    #df_train_loss = pd.read_csv(f"out/loss_{lr}_0.csv", header=None, names=["vals"])
    #df_train_acc = pd.read_csv(f"out/acc_{lr}_0.csv", header=None, names=["vals"])
    df_test_loss = pd.read_csv(f"out/0/test_loss_{lr}_20e.csv", header=None, names=["vals", "empty"])
    df_test_accu = pd.read_csv(f"out/0/test_acc_{lr}_20e.csv", header=None, names=["vals", "empty"])

    lr_label = format_label(lr)

    loss = df_test_loss["vals"]
    accu = df_test_accu["vals"]

    #plt.plot(df_train_loss["vals"], '.')
    ax1.plot(e, loss, '-o', linewidth=2, label=lr_label)

    #plt.plot(df_train_acc["vals"], '.')
    ax2.plot(e, accu, '-o', linewidth=2, label=lr_label)

    # zoom regions
    ax1_inset.plot(e, loss, '-o', linewidth=2)
    ax2_inset.plot(e, accu, '-o', linewidth=2)

  ax1.legend(fontsize=14)
  ax1.set_xlabel('Epochs', fontsize=16)
  ax1.set_ylabel('Loss', fontsize=16)
  ax1.tick_params(axis='both', which='major', labelsize=14)
  
  ax2.legend(fontsize=14)
  ax2.set_xlabel('Epochs', fontsize=16)
  ax2.set_ylabel('Accuracy', fontsize=16)
  ax2.tick_params(axis='both', which='major', labelsize=14)

  plt.show()

def main():
  lr_studies_dense()

if __name__ == "__main__":
  main()