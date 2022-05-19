# Import libraries
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np

# Define class names and import data from file
names = ['price', 'horsepower', 'luxury', 'efficiency', 'price per hp', 'weight', 'brand']
data = np.genfromtxt('input/data.csv', delimiter=',', dtype=None, encoding="utf-8", names=names)

# Define X and Y values
x = data['price']
y = data['horsepower']

# xSmooth = np.linspace(xPoints.min(), yPoints.max())
# spl = make_interp_spline(xPoints, yPoints, k=3)
# ySmooth = spl(xSmooth)
# plt.plot(xSmooth, ySmooth, label='smooth')

# Plot the result in a gridded graph
plt.plot(x, y, label="linear")
plt.grid()
plt.legend(loc='best')
plt.interactive(True)
plt.show()
