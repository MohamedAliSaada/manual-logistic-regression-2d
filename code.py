# =====================================================
# 1. Import libraries
# =====================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym

# =====================================================
# 2. Load and prepare the data
# =====================================================
df = pd.read_csv('circle_data_200.csv')
df.head()

# Features and labels
X = np.array(df[['position_x', 'position_y']])
Y = np.array(df['is_outside']).reshape(-1, 1)

# =====================================================
# 3. Feature engineering: add squared features
# =====================================================
X_t = np.c_[X, X**2]  # Add x1^2 and x2^2

# =====================================================
# 4. Initialize parameters
# =====================================================
w = np.zeros((X_t.shape[1], 1))  # (4,1) weights
b = 0

# =====================================================
# 5. Set hyperparameters
# =====================================================
m = X_t.shape[0]      # number of samples
epochs = 1000            # number of training epochs
alpha = 0.01          # learning rate

# =====================================================
# 6. Define cost function
# =====================================================
def cost_value(m, Y, y_pred):
    cost = -(1/m) * np.sum(Y * np.log(y_pred + 1e-15) + (1 - Y) * np.log(1 - y_pred + 1e-15))
    return cost

# =====================================================
# 7. Train the logistic regression model
# =====================================================
cost_history = []

for i in range(epochs):
    # Forward pass
    z = np.dot(X_t, w) + b
    y_pred = 1 / (1 + np.exp(-z))

    # Compute cost
    cost = cost_value(m, Y, y_pred)
    cost_history.append(cost)

    # Backward pass (gradients)
    db = (1/m) * np.sum(y_pred - Y)
    dw = (1/m) * np.dot(X_t.T, (y_pred - Y))

    # Update parameters
    b = b - alpha * db
    w = w - alpha * dw

# =====================================================
# 8. Visualize the original data
# =====================================================
plt.figure(figsize=(8,6))
plt.scatter(X[Y[:,0]==0][:,0], X[Y[:,0]==0][:,1], marker='o', color='k', label='Inside Circle')
plt.scatter(X[Y[:,0]==1][:,0], X[Y[:,0]==1][:,1], marker='+', color='r', label='Outside Circle')
plt.xlabel('position_x')
plt.ylabel('position_y')
plt.title('Training Data Distribution')
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# 9. Plot decision boundary with data (using contour)
# =====================================================

# Create a grid of x1 and x2 values
x1_vals = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 300)
x2_vals = np.linspace(X[:,1].min()-1, X[:,1].max()+1, 300)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Compute the decision function f(x1, x2) on the grid
F = w[0,0]*X1 + w[1,0]*X2 + w[2,0]*X1**2 + w[3,0]*X2**2 + b

# Plot the decision boundary
plt.figure(figsize=(8,6))
plt.contour(X1, X2, F, levels=[0], colors='blue')  # Draw f(x1,x2)=0
plt.scatter(X[Y[:,0]==0][:,0], X[Y[:,0]==0][:,1], marker='o', color='k', label='Inside Circle')
plt.scatter(X[Y[:,0]==1][:,0], X[Y[:,0]==1][:,1], marker='+', color='r', label='Outside Circle')
plt.xlabel('position_x')
plt.ylabel('position_y')
plt.title('Decision Boundary and Data Points')
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# 10. Plot decision boundary with data
# =====================================================
plt.figure(figsize=(8,6))
plt.scatter(X[Y[:,0]==0][:,0], X[Y[:,0]==0][:,1], marker='o', color='k', label='Inside Circle')
plt.scatter(X[Y[:,0]==1][:,0], X[Y[:,0]==1][:,1], marker='+', color='r', label='Outside Circle')

# Plot decision boundary
if points.shape[0] > 0:
    plt.plot(points[:,0], points[:,1], color='blue', label='Decision Boundary Points')

plt.xlabel('position_x')
plt.ylabel('position_y')
plt.title('Decision Boundary after Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# 11. Calculate and print model accuracy
# =====================================================
def accu_logistic(y_pred, Y):
    pred_int = (y_pred >= 0.5).astype(int)
    accu = np.mean(pred_int == Y)
    return f'The accuracy is {accu*100:.2f}%'

# Output the accuracy
print(accu_logistic(y_pred, Y))
