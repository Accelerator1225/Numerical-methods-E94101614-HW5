
import numpy as np

# Exact solution for Q1
def exact_solution_q1(t):
    return np.tan(np.log(t))

# Exact solution for Q2
def exact_solution_q2(t):
    u1 = np.exp(-3*t) - np.exp(-9*t) + np.cos(t)
    u2 = -np.exp(-3*t) + np.exp(-9*t) + np.cos(t)
    return u1, u2

# -------------------------
# Q1: Single ODE
# -------------------------
def question1():
    h = 0.1
    t_values = np.arange(1, 2 + h, h)
    n = len(t_values)

    def f(t, y):
        return y / t + y / t**2 + 1 / t**2

    def df(t, y):
        df_dt = -2 / t**3
        df_dy = 1 / t + 1 / t**2
        return df_dt + df_dy * f(t, y)

    y_euler = np.zeros(n)
    y_euler[0] = 0
    for i in range(n-1):
        y_euler[i+1] = y_euler[i] + h * f(t_values[i], y_euler[i])

    y_taylor = np.zeros(n)
    y_taylor[0] = 0
    for i in range(n-1):
        y_taylor[i+1] = y_taylor[i] + h * f(t_values[i], y_taylor[i]) + (h**2 / 2) * df(t_values[i], y_taylor[i])

    y_exact = exact_solution_q1(t_values)

    print("Q1: Single ODE")
    print("t\tEuler\t\tTaylor 2nd\tExact")
    for i in range(n):
        print(f"{t_values[i]:.1f}\t{y_euler[i]:.6f}\t{y_taylor[i]:.6f}\t{y_exact[i]:.6f}")

# -------------------------
# Q2: System of Two ODEs
# -------------------------
def question2():
    def f1(t, u1, u2):
        return (9*u1 + 24*u2 + 5*np.cos(t) - np.sin(t)) / 3

    def f2(t, u1, u2):
        return (-24*u1 - 52*u2 - 9*np.cos(t) + np.sin(t)) / 3

    def runge_kutta(h):
        t_values = np.arange(0, 1 + h, h)
        u1_values = np.zeros(len(t_values))
        u2_values = np.zeros(len(t_values))
        u1_values[0] = 4/3
        u2_values[0] = 2/3

        for i in range(len(t_values)-1):
            t = t_values[i]
            u1 = u1_values[i]
            u2 = u2_values[i]

            k1_u1 = h * f1(t, u1, u2)
            k1_u2 = h * f2(t, u1, u2)

            k2_u1 = h * f1(t + h/2, u1 + k1_u1/2, u2 + k1_u2/2)
            k2_u2 = h * f2(t + h/2, u1 + k1_u1/2, u2 + k1_u2/2)

            k3_u1 = h * f1(t + h/2, u1 + k2_u1/2, u2 + k2_u2/2)
            k3_u2 = h * f2(t + h/2, u1 + k2_u1/2, u2 + k2_u2/2)

            k4_u1 = h * f1(t + h, u1 + k3_u1, u2 + k3_u2)
            k4_u2 = h * f2(t + h, u1 + k3_u1, u2 + k3_u2)

            u1_values[i+1] = u1 + (k1_u1 + 2*k2_u1 + 2*k3_u1 + k4_u1) / 6
            u2_values[i+1] = u2 + (k1_u2 + 2*k2_u2 + 2*k3_u2 + k4_u2) / 6

        return t_values, u1_values, u2_values

    print("\nQ2: System of Two ODEs")
    for h in [0.05, 0.1]:
        t_values, u1_values, u2_values = runge_kutta(h)
        print(f"\nStep size h = {h}")
        print("t\tu1 (RK4)\tu1 (Exact)\tu2 (RK4)\tu2 (Exact)")
        for i in range(len(t_values)):
            u1_exact, u2_exact = exact_solution_q2(t_values[i])
            print(f"{t_values[i]:.2f}\t{u1_values[i]:.6f}\t{u1_exact:.6f}\t{u2_values[i]:.6f}\t{u2_exact:.6f}")

# -------------------------
# Main function
# -------------------------
if __name__ == "__main__":
    question1()
    question2()
