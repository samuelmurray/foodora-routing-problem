""" Functions to plot graphs of the SimAnn solution """

import matplotlib.pyplot as plt
import numpy as np
import run

problem, solver1 = run.run()
solver2 = run.runJustSA(problem)
solver3 = run.runJustSA(problem)
solvers = [solver1, solver2, solver3]

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)
plt.style.use('./myReportStyle.mplstyle')
# plt.style.use('ciwans-report')
sufix = "_smth"
plt.ioff()
fig1 = plt.figure()
ax1 = plt.subplot(111)
s = 0
for solver in solvers:
    data = solver.costData
    bestData = solver.bestCostData
    k = np.arange(0, len(data))
    # do plot

    ax1.plot(k, data, color=tableau20[s])
    ax1.hold(True)
    for i in range(len(bestData)):
        ax1.plot(solver.bestCostData[i][1], solver.bestCostData[i][0], '*', color='black', markeredgecolor='black')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.tight_layout(pad=0.5)
    plt.savefig("costPlot" + sufix + ".pdf")
    s += 1
plt.show()

fig2 = plt.figure()
ax2 = plt.subplot(111)
s = 0
for solver in solvers:
    data = solver.tempData
    bestData = solver.bestCostData
    k = np.arange(0, len(data))
    # do plot

    ax2.plot(k, data, color=tableau20[s])
    ax2.hold(True)
    #    for i in range(len(bestData)):
    #        kstar = solver.bestCostData[i][1]
    #        ax2.plot(kstar, data[kstar], '*',
    #                 color = tableau20[s + len(solvers)], markeredgecolor = 'black')
    plt.xlabel('Iterations')
    plt.ylabel('Temperature')
    plt.tight_layout(pad=0.5)
    plt.savefig("tempPlot" + sufix + ".pdf")
    s += 1
plt.show()
