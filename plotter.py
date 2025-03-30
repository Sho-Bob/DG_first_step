import matplotlib.pyplot as plt
import numpy as np

def plot(u,x_element,primitive_variable):
    x_coord = x_element.flatten()
    u_coord = u[:,:,0].flatten()
    p_coord = primitive_variable[:,:,2].flatten()

    plt.figure(figsize=(8, 6))
    plt.plot(x_coord, u_coord, linestyle='-', color='k', label='Density')
    plt.plot(x_coord, p_coord, linestyle='-', color='r', label='Pressure')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_cell_average(cons_v_cell_average,prim_v_cell_average):
    x_coord = np.arange(cons_v_cell_average.shape[0])
    plt.figure(figsize=(8, 6))
    plt.plot(x_coord, cons_v_cell_average[:,0], linestyle='-', color='k', label='Density')
    plt.plot(x_coord, prim_v_cell_average[:,2], linestyle='-', color='r', label='Pressure')
    plt.xlabel('x')
    plt.ylabel('Primitive Variables')
    plt.legend()