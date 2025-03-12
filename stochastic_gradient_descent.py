import csv

data = []
FEATURES = 5
sf = [0] * FEATURES
with open("house-prices.csv", mode="r") as file:
    reader = csv.reader(file)
    for row in reader:
        data.append([int(row[1]),int(row[2]),int(row[3]),int(row[4]),int(row[5])])
        for i in range(len(sf)):
            sf[i] = max(sf[i], int(row[i+1])) 

# scaling   
for i in range(len(data)):
    for j in range(len(data[0])):
        data[i][j] /= sf[j]

# hypothesis
theta = [0] * (FEATURES)
def h(x: list[int]) -> float:
    s = 0
    for i in range(FEATURES - 1):
        s += x[i] * theta[i]
    s += theta[-1]
    return s

# total loss function
def j() -> float:
    l = 0
    for i in range(len(data)):
        l += (h(data[i][1:]) - data[i][0]) ** 2
    return l/len(data)

# stochastic gradient descent
def stochastic(theta : list[int], alpha : float) -> float:
    for i in range(len(data)):
        e = h(data[i][1:]) - data[i][0]
        theta[-1] = theta[-1] - alpha * e
        for p in range(1, FEATURES):
            theta[p-1] = theta[p-1] - alpha * e * data[i][p]
    return j()

tolerance = 0.0000001
prev_cost = float('inf')

while True:
    c = stochastic(theta, 0.01)
    if (abs(c-prev_cost) < tolerance):
        break
    prev_cost = c

f = [1790,2,2,2] # SqFt, Bedrooms, Bathrooms, Offers

# scale features
for i in range(len(f)):
    f[i] /= sf[i+1]

print(f"Predicted Price: ${(h(f)*sf[0]):.2f}")