import numpy as np
import matplotlib.pyplot as plt

# load data
with open('LBFGS_log.txt', 'r') as file:
    data = file.read()


# Initialize arrays
optimal_x = []
update_history = []
true_optimal_x = []

# Split the input text into lines
lines = data.strip().split('\n')

# Initialize a variable to keep track of the current array
current_array = []

# Iterate through each line
for line in lines:
    line = line.strip()
    
    if line == "The optimal x is:":
        optimal_x.append([])
        current_array = optimal_x[-1]
        continue

    elif line == "The update history is:":
        update_history.append([])
        current_array = update_history[-1]
        continue

    elif line == "The true optimal x is:":
        true_optimal_x.append([])
        current_array = true_optimal_x[-1]
        continue

    else:
        current_array.append(float(line))


test_num = len(optimal_x)

# Check the correctness of LBFGS implementation
correct = True
for i in range(test_num-2):
    if not np.allclose(optimal_x[i], true_optimal_x[i], atol=1e-6):
        correct = False
        break
print(correct)

# Plot the convergence curve in one figure with five subplots
fig, axs = plt.subplots(5, 1, figsize=(10, 20))
for i in range(test_num):
    axs[i].plot(update_history[i], label='Test %d, diff of x_k and optimal' % i)
    axs[i].set_title('Convergence curve')
    axs[i].set_xlabel('Iteration')
    axs[i].set_ylabel('Diff between x_k and optimal')
    axs[i].legend()

plt.tight_layout()
plt.savefig('convergence_curve.png')
plt.show()











