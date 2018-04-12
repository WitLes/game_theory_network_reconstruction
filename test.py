import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import  evolutionary_game
a = np.matrix([1,0])
b = np.matrix([[1,2],[3,4]])
c = np.matrix([0,1])
d = np.transpose(c)
print(a,d)
a = evolutionary_game.multi_dot(a,b,d)
print(a.item())

