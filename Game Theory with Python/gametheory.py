# !pip install nashpy==0.0.19
# !pip install axelrod

import nashpy as nash
import numpy as np

# Create the payoff matrix

P1 = np.array([[8, 1], [15, 3]])  # P1 is the row player
P2 = np.array([[8, 15], [1, 3]])  # P2 is the column player
pd = nash.Game(P1, P2)
print(pd)

# Exercise
# A = np.array([[5,17],[14,12]])
# B = np.array([[15,16],[2,8]])
# gm = nash.Game(A,B)
# print(gm)

# Calculate Utilities

sigma_r = np.array([.2, .8])
sigma_c = np.array([.6, .4])
pd = nash.Game(P1, P2)
print(pd[sigma_r, sigma_c])

# ur(sr,sc)
# uc(sr,sc)
ur = 0.2*0.6*8+0.2*0.4*1+0.8*0.6*15+0.8*0.4*3
uc = 0.2*0.6*8+0.2*0.4*15+0.8*0.6*1+0.8*0.4*3
print(ur)
print(uc)

#  Exercise: Calculate the utilities of the game 'gm' created in the previous exercise, using: sr=(.3,.7) and sc=(.5,.5);
# sigma_r = np.array([.3,.7])
# sigma_c = np.array([.5,.5])
# gm = nash.Game(A, B)
# gm[sigma_r, sigma_c]

# Find the Nash Equilibrium with Support Enumeration

equilibria = pd.support_enumeration()
for eq in equilibria:
  print(eq)

# equilibria = gm.support_enumeration()
# for eq in equilibria:
#   print(eq)

P3 = np.array([[3, 1], [4, 0]])  # P3 is the row player
P4 = np.array([[3, 4], [1, 0]])  # P4 is the column player
hd = nash.Game(P3, P4)
print(hd)

equilibria = hd.support_enumeration()
for eq in equilibria:
    print(eq)

# M = np.array([[1, 1, 3, 2], [2, 3, 4, 3], [5, 1, 1, 4 ]])
# N = np.array([[3, 2, 2, 4], [1, 4, 2, 0], [3, 3, 2, 3]])
# mn = nash.Game(M, N)
# print(mn)
# equilibria = mn.support_enumeration()
# for eq in equilibria:
#   print(eq)

P5 = np.array([[1, -1], [-1, 1]])
mp = nash.Game(P5)
print(mp)

equilibria = mp.support_enumeration()
for eq in equilibria:
  print(eq)

# Z1 = np.array([[5, -6.5], [-2.5, 7]])
# zs = nash.Game(Z1)
# print(zs)
# equilibria = zs.support_enumeration()
# for eq in equilibria:
#   print(eq)

#!pip install -U pyYAML     # Troubleshoot: Execute this line if Axelrod does not run and AttributeError: module 'yaml' has no attribute 'FullLoader' occurs
# Import package

import axelrod as axl

# Create matches
players = (axl.Cooperator(), axl.Alternator())                 # using players of Cooperator and Alternator strategy
match1 = axl.Match(players, turns=5)                           # play for 5 turns
print(match1.play())

# print(axl.all_strategies) # shows possible strategies
# players = (axl.TitForTat(), axl.rand.Random())
# match2 = axl.Match(players, turns=15)
# print(match2.play())
# Payoffs

print(match1.game)        #Analyze the match
#These payoffs are commonly referred to as:

#R: the Reward payoff (default value in the library: 3) C-C
#P: the Punishment payoff (default value in the library: 1) D-D
#S: the Loss payoff (default value in the library: 0) C-D
#T: the Temptation payoff (default value in the library: 5) D-C
# Scores of a match
print(match1.scores())    #Retrieve match scores
# The result of the match can also be viewed as sparklines where cooperation is shown as a solid block and defection as a space.
print(match1.sparklines())  # Get output using sparklines
