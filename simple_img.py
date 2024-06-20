import matplotlib.pyplot as plt
import numpy as np

def main():
  t = np.linspace(0, 10*2*np.pi, 512)

  w = 0.5
  phi = 0

  y = np.sin(w*t + phi)
  yn = np.sin(w*t + phi) + np.random.normal(0, 0.1, 512)

  plt.figure()
  plt.plot(t, yn)
  plt.plot(t, y)
  plt.show()

if __name__ == "__main__":
  main()