
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


# 1. Load the dataset
url = "https://raw.githubusercontent.com/SharifiZarchi/IntroAI/main/Session_04/TehranHouses/housePrice.csv"

df = pd.read_csv(url)

print("Original dataset size:", len(df))


# 2. Clean the data
df["Price"] = (
    df["Price"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Area"] = pd.to_numeric(df["Area"], errors="coerce")

df = df.dropna(subset=["Area", "Price"])

# Remove invalid values
df = df[(df["Area"] > 0) & (df["Price"] > 0)]

print("Clean dataset size:", len(df))


# 3. Prepare features and target
X = df[["Area"]]
y = df["Price"] / 1_000_000_000

print("X shape:", X.shape)
print("y shape:", y.shape)


# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# 5. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")
print("Slope:", round(model.coef_[0], 4))
print("Intercept:", round(model.intercept_, 4))


# 6. Evaluate the model
predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("\nModel evaluation")
print("R2:", round(r2, 3))
print("MAE:", round(mae, 3), "billion Toman")


# 7. Predict the price of a 100 m2 apartment
new_apartment = pd.DataFrame({"Area": [100]})

prediction = model.predict(new_apartment)

print(
    "\nPredicted price for 100 m2:",
    round(prediction[0], 2),
    "billion Toman"
)


# 8. Plot real prices and the regression line
plt.figure(figsize=(9, 6))

plt.scatter(
    X_test["Area"],
    y_test,
    alpha=0.4,
    label="Real prices"
)

line_x = pd.DataFrame({
    "Area": range(
        int(df["Area"].min()),
        int(df["Area"].max()) + 1
    )
})

plt.plot(
    line_x["Area"],
    model.predict(line_x),
    color="red",
    linewidth=2,
    label="Linear regression"
)

plt.xlabel("Area (m2)")
plt.ylabel("Price (billion Toman)")
plt.title("Tehran House Price Prediction")
plt.legend()
plt.grid(alpha=0.2)

plt.tight_layout()
plt.show()
