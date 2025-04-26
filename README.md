# Logistic Regression (Manual Implementation) for 2D Binary Classification

This project manually implements a logistic regression model from scratch (without using machine learning libraries like scikit-learn or TensorFlow) to solve a binary classification problem based on 2D feature data.

---

## 🛠 Project Overview

- **Goal**: Build and train a logistic regression model using NumPy to classify 2D points into two classes.
- **Features**: `position_x` and `position_y` (2D coordinates).
- **Target**: Binary label `0` or `1`.
- **Techniques**:
  - Feature engineering with polynomial terms (`x1²`, `x2²`) to enable non-linear decision boundaries.
  - Manual implementation of gradient descent.
  - Visualization of both training data and the learned decision boundary.

---

## 🧪 Output Examples

### 📌 Training Data Distribution

Shows the distribution of the 2D points labeled as class 0 and class 1.

![Training Data Distribution](Training%20Data%20Distribution.png)

---

### 📌 Decision Boundary and Data Points

The trained logistic regression model learns a non-linear boundary.  
The plot shows how it separates the two classes based on the learned function.

![Decision Boundary and Data Points](Decision%20Boundary%20and%20Data%20Points.png)

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `logistic_regression_2d.py` | Main training and plotting script |
| `circle_data_200.csv` | Input dataset with two features and a binary label |
| `Training Data Distribution.png` | Output plot showing labeled training data |
| `Decision Boundary and Data Points.png` | Output plot showing learned decision boundary |

---

## 🚀 How to Run

1. Clone this repository.
2. Make sure `circle_data_200.csv` is in the root directory.
3. Run the script:

```bash
python logistic_regression_2d.py
