import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import ScalarFormatter

def lr_studies_dense_renplex():
  seed = 8912985653
  epochs = 32
  model_id = 0
  lr_list = [
    "1.50_0.00",
    "1.50_0.09",
    "1.48_0.23",
    "1.30_0.75",
    "1.06_1.06",
    "0.75_1.30",
  ]

  def format_label(x: str):
    x_new = x.replace("_", " + i")

    return fr"${x_new}$"
  
  f1 = plt.figure(figsize=(6.5, 6.5))
  f2 = plt.figure(figsize=(6.5, 6.5))
  ax1 = f1.add_subplot(111)
  ax2 = f2.add_subplot(111)
  
  # loss and accuracy zoom region
  e1, e2 = int(0.8 * epochs), int(1.05 * epochs)
  l1, l2 = 0.004, 0.010
  a1, a2 = 0.95, 0.975

  x_pos = -0.05
  ax1_inset = inset_axes(
    ax1,
    width="33%", height="33%",
    # (left, bottom, width, height)
    bbox_to_anchor=(x_pos, -0.45, 1.0, 1.0), bbox_transform=ax1.transAxes
  )
  ax1_inset.set_xlim(e1, e2)
  ax1_inset.set_ylim(l1, l2)
  ax1_inset.tick_params(axis='both', which='major', labelsize=12)


  ax2_inset = inset_axes(
    ax2,
    width="33%", height="33%",
    # (left, bottom, width, height)
    bbox_to_anchor=(x_pos, -0.18, 1.0, 1.0), bbox_transform=ax2.transAxes
  )
  ax2_inset.set_xlim(e1, e2)
  ax2_inset.set_ylim(a1, a2)
  ax2_inset.tick_params(axis='both', which='major', labelsize=12)

  e = list(range(1,epochs+1))
  for lr in lr_list:
    #df_train_loss = pd.read_csv(f"out/loss_{lr}_0.csv", header=None, names=["vals"])
    #df_train_acc = pd.read_csv(f"out/acc_{lr}_0.csv", header=None, names=["vals"])
    df_test_loss = pd.read_csv(f"out/{model_id}/{seed}_tloss_{lr}_{epochs}e.csv", header=None, names=["vals", "empty"])
    df_test_accu = pd.read_csv(f"out/{model_id}/{seed}_tacc_{lr}_{epochs}e.csv", header=None, names=["vals", "empty"])

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

def lr_studies_dense_tensorflow():
  seed = 2375973
  epochs = 32
  model_id = 0
  lr_list = [
    "3.0",
    "5.0",
    "6.0",
    "7.0"
  ]
  
  f1 = plt.figure(figsize=(6.5, 6.5))
  f2 = plt.figure(figsize=(6.5, 6.5))
  ax1 = f1.add_subplot(111)
  ax2 = f2.add_subplot(111)
  
  # loss and accuracy zoom region
  e1, e2 = int(0.8 * epochs), int(1.05 * epochs)
  l1, l2 = 0.0045, 0.010
  a1, a2 = 0.94, 0.970

  x_pos = -0.05
  ax1_inset = inset_axes(
    ax1,
    width="33%", height="33%",
    # (left, bottom, width, height)
    bbox_to_anchor=(x_pos, -0.45, 1.0, 1.0), bbox_transform=ax1.transAxes
  )
  ax1_inset.set_xlim(e1, e2)
  ax1_inset.set_ylim(l1, l2)
  ax1_inset.tick_params(axis='both', which='major', labelsize=12)


  ax2_inset = inset_axes(
    ax2,
    width="33%", height="33%",
    # (left, bottom, width, height)
    bbox_to_anchor=(x_pos, -0.18, 1.0, 1.0), bbox_transform=ax2.transAxes
  )
  ax2_inset.set_xlim(e1, e2)
  ax2_inset.set_ylim(a1, a2)
  ax2_inset.tick_params(axis='both', which='major', labelsize=12)

  e = list(range(1,epochs+1))
  for lr in lr_list:
    #df_train_loss = pd.read_csv(f"out/loss_{lr}_0.csv", header=None, names=["vals"])
    #df_train_acc = pd.read_csv(f"out/acc_{lr}_0.csv", header=None, names=["vals"])
    df_test_loss = pd.read_csv(f"out/{model_id}/{seed}_id{model_id}_tloss_{lr}.csv")
    df_test_accu = pd.read_csv(f"out/{model_id}/{seed}_id{model_id}_taccu_{lr}.csv")

    loss = df_test_loss["vals"]
    accu = df_test_accu["vals"]

    #plt.plot(df_train_loss["vals"], '.')
    ax1.plot(e, loss, '-o', linewidth=2, label=lr)

    #plt.plot(df_train_acc["vals"], '.')
    ax2.plot(e, accu, '-o', linewidth=2, label=lr)

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

def comp():
  seed_list = [
    891298565,
    918232853, # 435726692
    328557473,
    348769349,
    224783561,
    981347827
  ]

  seed_tensorflow = [
    891298565,
    435726692,
    328557473,
    348769349,
    224783561,
    981347827
  ]

  epochs = 32
  model_id = 3
  lr_renplex = "0.0350_0.0011"
  lr_tensorflow = "17.5"

  # setting up plot axes
  f1 = plt.figure(figsize=(6.5, 6.5))
  f2 = plt.figure(figsize=(6.5, 6.5))
  ax1 = f1.add_subplot(111)
  ax2 = f2.add_subplot(111)
  
  # loss and accuracy zoom region
  e1, e2 = int(0.8 * epochs), int(1.05 * epochs)
  l1, l2 = 0.01, 0.03
  a1, a2 = 0.975, 0.990

  x_pos = -0.05
  ax1_inset = inset_axes(
    ax1,
    width="33%", height="33%",
    # (left, bottom, width, height)
    bbox_to_anchor=(x_pos, -0.45, 1.0, 1.0), bbox_transform=ax1.transAxes
  )
  ax1_inset.set_xlim(e1, e2)
  ax1_inset.set_ylim(l1, l2)
  ax1_inset.tick_params(axis='both', which='major', labelsize=12)


  ax2_inset = inset_axes(
    ax2,
    width="33%", height="33%",
    # (left, bottom, width, height)
    bbox_to_anchor=(x_pos, -0.18, 1.0, 1.0), bbox_transform=ax2.transAxes
  )
  ax2_inset.set_xlim(e1, e2)
  ax2_inset.set_ylim(a1, a2)
  ax2_inset.tick_params(axis='both', which='major', labelsize=12)

  e = list(range(1, epochs+1))
  e_ticks = list(range(1, epochs+1, 4))

  # getting values
  loss_vals = []
  #accu_vals = []
  loss_tf_vals = []
  #accu_tf_vals = []
  for seed, seed_tf in zip(seed_list, seed_tensorflow):
    df_test_loss = pd.read_csv(f"out/{model_id}/{seed}_tloss_{lr_renplex}_{epochs}e.csv", header=None, names=["vals", "empty"])
    #df_test_accu = pd.read_csv(f"out/{model_id}/{seed}_taccu_{lr_renplex}_{epochs}e.csv", header=None, names=["vals", "empty"])
    df_test_loss_tf = pd.read_csv(f"tensorflow/{seed_tf}_id{model_id}_tloss_{lr_tensorflow}_{epochs}e.csv")
    #df_test_accu_tf = pd.read_csv(f"tensorflow/{seed_tf}_id{model_id}_taccu_{lr_tensorflow}_{epochs}e.csv")
    
    loss = df_test_loss["vals"]
    #accu = df_test_accu["vals"]
    loss_tf = df_test_loss_tf["vals"]
    #accu_tf = df_test_accu_tf["vals"]

    loss_vals.append(loss)
    #accu_vals.append(accu)
    loss_tf_vals.append(loss_tf)
    #accu_tf_vals.append(accu_tf)

  loss_array = np.array(loss_vals)
  loss_tf_array = np.array(loss_tf_vals)
  #accu_array = np.array(accu_vals)
  #accu_tf_array = np.array(accu_tf_vals)

  mean_loss = np.mean(loss_array, axis=0)
  mean_loss_tf = np.mean(loss_tf_array, axis=0)
  #mean_accu = np.mean(accu_array, axis=0)
  #mean_accu_tf = np.mean(accu_tf_array, axis=0)

  std_loss = 2 * np.std(loss_array, axis=0)
  std_loss_tf = 2 * np.std(loss_tf_array, axis=0)
  #std_accu = 2 * np.std(accu_array, axis=0)
  #std_accu_tf = 2 * np.std(accu_tf_array, axis=0)

  line_width = 2.5
  trans = 0.3
  ax1.plot(e, mean_loss, '--', color="blue", linewidth=line_width, label=r"Renplex $\mu$")
  ax1.plot(e, mean_loss_tf, '--', color="orange", linewidth=line_width, label=r"Tensorflow $\mu$")
  ax1.fill_between(e, mean_loss-std_loss, mean_loss+std_loss, color='blue', alpha=trans, label=r'Renplex $\mu \pm 2\sigma$')
  ax1.fill_between(e, mean_loss_tf-std_loss_tf, mean_loss_tf+std_loss_tf, color='orange', alpha=trans, label=r'Tensorflow $\mu \pm 2\sigma$')
  # zoom regions
  ax1_inset.plot(e, mean_loss, '--', color="blue", linewidth=line_width)
  ax1_inset.plot(e, mean_loss_tf, '--', color="orange", linewidth=line_width)
  ax1_inset.fill_between(e, mean_loss-std_loss, mean_loss+std_loss, color='blue', alpha=trans)
  ax1_inset.fill_between(e, mean_loss_tf-std_loss_tf, mean_loss_tf+std_loss_tf, color='orange', alpha=trans)


  #ax2.plot(e, mean_accu, '--', color="blue", linewidth=line_width, label=r"Renplex $\mu$")
  #ax2.plot(e, mean_accu_tf, '--', color="orange", linewidth=line_width, label=r"Tensorflow $\mu$")
  #ax2.fill_between(e, mean_accu-std_accu, mean_accu+std_accu, color='blue', alpha=trans, label=r'Renplex $\mu \pm 2\sigma$')
  #ax2.fill_between(e, mean_accu_tf-std_accu_tf, mean_accu_tf+std_accu_tf, color='orange', alpha=trans, label=r'Tensorflow $\mu \pm 2\sigma$')
  # zoom regions
  #ax2_inset.plot(e, mean_accu, '--', color="blue", linewidth=line_width)
  #ax2_inset.plot(e, mean_accu_tf, '--', color="orange", linewidth=line_width)
  #ax2_inset.fill_between(e, mean_accu-std_accu, mean_accu+std_accu, color='blue', alpha=trans)
  #ax2_inset.fill_between(e, mean_accu_tf-std_accu_tf, mean_accu_tf+std_accu_tf, color='orange', alpha=trans)

  ax1.legend(fontsize=14, loc=1)
  ax1.set_xlabel('Epochs', fontsize=16)
  ax1.set_ylabel('Loss', fontsize=16)
  ax1.tick_params(axis='both', which='major', labelsize=14)
  ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
  ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
  ax1.set_xticks(e_ticks)
  offset_text = ax1.yaxis.get_offset_text()
  offset_text.set_fontsize(14)
  
  ax2.legend(fontsize=14, loc=4)
  ax2.set_xlabel('Epochs', fontsize=16)
  ax2.set_ylabel('Accuracy', fontsize=16)
  ax2.tick_params(axis='both', which='major', labelsize=14)
  ax2.set_xticks(e_ticks)

  plt.show()

def main():
  comp()

if __name__ == "__main__":
  main()