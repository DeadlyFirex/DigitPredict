# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

# Define class names and import data from file
names = ['price', 'horsepower', 'luxury', 'efficiency', 'price per hp', 'weight', 'brand']
dataset = read_csv('input/data.csv', names=names)

# Printout data
print("━━━━━━━━━━━━┫ START ┣━━━━━━━━━━━━")
print(dataset.shape)
print("──────────────────────────────────")
print(dataset.head(20))
print("──────────────────────────────────")
print(dataset.describe())
print("──────────────────────────────────")
print(dataset.groupby('class').size())
print("━━━━━━━━━━━━┫ END ┣━━━━━━━━━━━━")

# Define layout and plot all graphs, save them into output/
dataset.plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False)
pyplot.savefig('output/boxplot.pdf', dpi=1200)
dataset.hist()
pyplot.savefig('output/histogram.pdf', dpi=1200)
scatter_matrix(dataset)
pyplot.savefig('output/scatter.pdf', dpi=1200)
pyplot.interactive(True)
pyplot.show()