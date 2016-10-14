To run the PDDL planner fast downward has to be download and installed. Unfortunately it is not very well supported for windows but for Mac and Linux there should be no major problems. Some additional dependencies has to be installed as well. They are all listed under dependencies along with instructions on how to install the program here:
http://www.fast-downward.org/ObtainingAndRunningFastDownward

The file run.py runs the simulated annealing for a problem specified by three different .json files opened on row 9, 12 and 15 in the code. There is only one graph but we have several different test cases for the number of bikers and costumers inside the data folder. The script also outputs a file in the data folder named pddl_init.pddl. This is used to run fast downward from the terminal with the command:
./fast-downward.py foodora_cost_domain.pddl pddl_init.pddl --search "astar(lmcut())"

Note that the output "Plan Cost" needs to be divided by 1000 to give the cost corresponding to the distances used in the simulated annealing. 

To retrieve plots over the simulated annealing run plotMCMCresult.py. This will also give an output file named plotData that contains the bestCost followed by the cost for all the different routes. The cost function is defined and may be changed at row 222 in the file MCMC.py to test different cases. The combined case is retrieved by writing:

return np.sum(routeCosts) + self.nrBikes*np.amax(routeCosts) 