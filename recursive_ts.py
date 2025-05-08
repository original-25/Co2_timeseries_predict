import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import legend
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
data = pd.read_csv("co2.csv")

data["time"] = pd.to_datetime(data["time"])

# fig, ax = plt.subplots()
# ax.plot(data["time"], data["co2"], label="Co2")
# plt.show()

data["co2"] = data["co2"].interpolate()


# fig, ax = plt.subplots()
# ax.plot(data["time"], data["co2"], label="Co2")
# plt.show()

def create_ts_data(data, window_size=5):
    i = 1
    while i < window_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i+=1
    data["target"] = data["co2"].shift(-i)
    data.dropna(axis="index", inplace=True)
    return data

data = create_ts_data(data)


x = data.drop(["target", "time"], axis="columns")
y = data["target"]

train_ratio = 0.8

x_train = x[:int(len(x)*train_ratio)]
y_train = y[:int(len(y)*train_ratio)]

x_test = x[int(len(x)*train_ratio):]
y_test = y[int(len(y)*train_ratio):]

reg = LinearRegression()

reg.fit(x_train, y_train)

y_predicted = reg.predict(x_test)

print("MAE: {}".format(mean_absolute_error(y_test, y_predicted)))
print("MSE: {}".format(mean_squared_error(y_test, y_predicted)))
print("R2 score: {}".format(r2_score(y_test, y_predicted)))

# plt.plot(data.time[:int(len(x)*train_ratio)], data["co2"][:int(len(x)*train_ratio)], label="train")
# plt.plot(data.time[int(len(x)*train_ratio):], data["co2"][int(len(x)*train_ratio):], label="test")
# plt.plot(data.time[int(len(x)*train_ratio):], y_predicted, label="predicted")
# plt.grid()
# plt.legend()
# plt.show()

test_sample = [380, 381, 382, 385, 391]
prediction = reg.predict([test_sample])

for i in range(10):
    prediction = reg.predict([test_sample])
    print("Input is: ", test_sample)
    print(prediction[0])
    print("___________________________________________")
    test_sample.pop(0)
    test_sample.append(prediction[0])