import sys
sys.path.append('../../src')
import dg_tools as dg

import numpy as np
import matplotlib.pyplot as plt

"""
    
"""

print("Testing angles between unit vectors:")

atom     = np.array([1,0])
vertex   = np.array([0,0])

plt.plot(atom,vertex, 'k-')
y_cord   = np.linspace(0.0, 1.0, 10)
# between 0 and 90:
for y in y_cord:
    x  = np.sqrt(1 - y**2)
    cv = np.array([x,y])
    plt.plot(cv[0],cv[1],'r*')
    print(dg.get_angle(atom, vertex, cv))

# between 90 and 180:
for y in np.flip(y_cord):
    x  = -1*np.sqrt(1 - y**2)
    cv = np.array([x,y])
    plt.plot(cv[0],cv[1],'b*')
    print(dg.get_angle(atom, vertex, cv))

# between 180 and 270:
y_cord = -1.0 * np.flip(y_cord)
for y in np.flip(y_cord):
    x  = -1*np.sqrt(1 - y**2)
    cv = np.array([x,y])
    plt.plot(cv[0],cv[1],'g*')
    print(dg.get_angle(atom, vertex, cv))

# between 270 and 360:
for y in y_cord:
    x  = np.sqrt(1 - y**2)
    cv = np.array([x,y])
    plt.plot(cv[0],cv[1],'m*')
    print(dg.get_angle(atom, vertex, cv))

plt.show()
print()
print("Testing angles between non-unite vectors")


plt.plot(atom,vertex, 'k-')
y_cord   = np.linspace(0.0, 3.3, 10)
# between 0 and 90:
for y in y_cord:
    x  = np.sqrt(3.3**2 - y**2)
    cv = np.array([x,y])
    plt.plot(cv[0],cv[1],'r*')
    print(dg.get_angle(atom, vertex, cv))

# between 90 and 180:
for y in np.flip(y_cord):
    x  = -1*np.sqrt(3.3**2 - y**2)
    cv = np.array([x,y])
    plt.plot(cv[0],cv[1],'b*')
    print(dg.get_angle(atom, vertex, cv))

# between 180 and 270:
y_cord = -1.0 * np.flip(y_cord)
for y in np.flip(y_cord):
    x  = -1*np.sqrt(3.3**2 - y**2)
    cv = np.array([x,y])
    plt.plot(cv[0],cv[1],'g*')
    print(dg.get_angle(atom, vertex, cv))

# between 270 and 360:
for y in y_cord:
    x  = np.sqrt(3.3**2 - y**2)
    cv = np.array([x,y])
    plt.plot(cv[0],cv[1],'m*')
    print(dg.get_angle(atom, vertex, cv))

plt.show()

