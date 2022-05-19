# Load libraries
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# Define class names and import data from file
names = ['price', 'horsepower', 'luxury', 'efficiency', 'price per hp', 'weight', 'brand']
dataset = read_csv('input/data.csv', delimiter=',', names=names, dtype=None, encoding="utf-8")

# Split the data into two variables
array = dataset.values
x = array[:, 0:6]
y = array[:, 6]

# Split the data for training / validation
X_train, x_placeholder, Y_train, y_placeholder = train_test_split(x, y, test_size=0.20, random_state=1, shuffle=True)
del x_placeholder, y_placeholder

# Define model
model = GaussianNB()
model.fit(X_train, Y_train)
guess = [[2500.0, 108.0, 1.35, 18.0, 5.1, 1000.0]]
prediction = model.predict(guess)

# Evaluate predictions
print(f"""
┌─────────────────────────┐
│Price: {guess[0][0]}
│HP: {guess[0][1]}
│Luxury: {guess[0][2]}
│Efficiency: {guess[0][3]}
│P/HP: {guess[0][4]}
│Weight: {guess[0][5]}
├─────────────────────────┤
│Result: {prediction[0]}
└─────────────────────────┘
""")
