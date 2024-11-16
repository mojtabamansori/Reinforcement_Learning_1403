import numpy as np
import matplotlib.pyplot as plt


#################### meghdar dhi avalie ###################################
gamma = 0.9
epsilon = 1e-5
P = np.array([[0.5, 0.3, 0.2, 0.0],
              [0.1, 0.6, 0.2, 0.1],
              [0.0, 0.2, 0.5, 0.3],
              [0.3, 0.0, 0.4, 0.3]])
R = np.array([2, 5, -1, 8])
V = np.zeros(len(P))
all_V = [np.copy(V)]
iteration = 0
done = False

############################# loop mohasbe #############################
while not done:
    delta = 0
    new_V = np.copy(V)
    print(f"Iteration {iteration}:")

    for s in range(len(P)):
        value = R[s] + gamma * np.sum(P[s, :] * V)
        print(f"  State {s}: V_old = {V[s]:.4f}, V_new = {value:.4f}")
        new_V[s] = value
        delta = max(delta, abs(new_V[s] - V[s]))

    V = new_V
    all_V.append(np.copy(V))
    iteration += 1
    print(f"  Delta: {delta:.6f}\n")

    if delta < epsilon:
        done = True

print("Final V:", V)

##############################  plot v graphic ##########################################
plt.figure(figsize=(10, 6))
all_V = np.array(all_V)
for i in range(len(P)):
    plt.plot(all_V[:, i], label=f'State {i}', marker='o')

plt.title('Value Iteration Progress for Each State')
plt.xlabel('Iteration')
plt.ylabel('Value (V)')
plt.legend()
plt.grid(True)
plt.savefig('mojtaba_mansori_nejad_1.png', dpi=300)
plt.show()

