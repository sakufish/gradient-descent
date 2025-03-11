import csv

data = []
m1, m2 = 0, 0
with open("house-prices.csv", mode="r") as file:
    reader = csv.reader(file)
    for row in reader:
        data.append([int(row[2]),int(row[1])])
        m1 = max(m1, int(row[1]))
        m2 = max(m2, int(row[2]))

# scaling
for i in range(len(data)):
    data[i][0] /= m2
    data[i][1] /= m1

# hypothesis
theta = [0,0]
def h(x: float) -> float:
    return theta[0] + x * theta[1]

# total loss function
def j() -> float:
    l = 0
    for i in range(len(data)):
        l += (h(data[i][0]) - data[i][1]) ** 2
    return l/len(data)

# gradient constant loss function 
def jgc() -> float:
    l = 0
    for i in range(len(data)):
        l += h(data[i][0]) - data[i][1]
    return l

# gradient term loss function
def jgt() -> float:
    l = 0
    for i in range(len(data)):
        l += (h(data[i][0]) - data[i][1]) * data[i][0]
    return l

# gradient descent - updates thetas and returns average cost
def gradient_descent(theta, alpha) -> float:
    theta[0] = theta[0] - alpha * jgc() / len(data)
    theta[1] = theta[1] - alpha * jgt() / len(data)
    return j()


prev_cost = float('inf')
tolerance = 0.00000001

while True:
    c = gradient_descent(theta, 0.01)
    if (abs(c-prev_cost) < tolerance):
        break
    prev_cost = c

print("Predicted Price for 2000 SqFt: " + str(h(2000/m2)*m1))