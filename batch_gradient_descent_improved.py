import csv

# Setup
data = []
sf1 = 0
sf2 = 0

with open("house-prices.csv", mode="r") as file:
    reader = csv.reader(file)
    for row in reader:
        price = int(row[1])
        sqft = int(row[2])
        data.append([price, sqft])
        sf1 = max(sf1, sqft)
        sf2 = max(sf2, price)

# scaling the data
for i in range(len(data)):
    data[i][1] /= sf1
    data[i][0] /= sf2 

# hypothesis
theta = [0, 0]  # [theta1, theta0]
def h(x):
    return x * theta[0] + theta[1]

# total loss function
def j():
    l = 0
    for i in range(len(data)):
        l += (h(data[i][1]) - data[i][0]) ** 2
    cost = l / (len(data)*2)
    print(cost)
    return cost

# gradient term for theta1
def jg1():
    l = 0
    for i in range(len(data)):
        l += (h(data[i][1]) - data[i][0]) * data[i][1]
    return l

# gradient term for theta0 (bias)
def jg0():
    l = 0
    for i in range(len(data)):
        l += h(data[i][1]) - data[i][0]
    return l

# gradient descent
def gradient_descent(alpha):
    global theta
    theta[0] -= alpha * jg1() / len(data)
    theta[1] -= alpha * jg0() / len(data)
    return j()

prev_cost = float('inf')
tolerance = 1e-7
alpha = 0.01

while True:
    cost = gradient_descent(alpha)
    if abs(cost - prev_cost) < tolerance:
        break
    prev_cost = cost

# scale to match model scale factor
test_sqft = 2000 / sf1
predicted_price = h(test_sqft) * sf2  # rescale to original price
print(predicted_price)