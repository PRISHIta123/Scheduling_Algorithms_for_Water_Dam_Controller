# Scheduling_Algorithms_for_Water_Dam_Controller

This repository contains code for the paper: An Intelligent RL-based Scheduler to Control Flooding in a Renewable Energy powered Automatic Water Dam control system (P.Ray, G.B. Amali- IEEE International Conference on Artificial Intelligence & Green Energy, 2023)

**Instructions to run:**  
1. Clone the project repository by downloading the zip file above or by using:    
```git clone https://github.com/PRISHIta123/Scheduling_Algorithms_for_Water_Dam_Controller.git```  
2. Navigate to the folder for a specific algorithm (choose from SAC_ERE_PER, PPO and NES)- an example is provided below:  
```cd/src/SAC_ERE_PER```  
3. Run the main python file  
```python main.py```
4. The algorithm-specific log files will be generated in the logs folder and the average rewards plot will be generated under the main plots folder.   

**To visualize more plots:** 
1. Run ```python combined_plot.py``` to generate the combined average rewards plot for all three algorithms (make sure to add the rewards output by the codes after running to get the plots for yourself, current reward values provided are from the run on my system).   
2. Similarly, run SEPS_States_plot.py, SEPS_Action_plot.py, PPO_States_Plot.py, PPO_Actions_Plot.py to visualize the states and actions for the last interaction (copy the states/actions for the last interaction in a similar 2D-list format from the respective log files similar to that of the previous run on my system).  

**To get the pairwise Mann-Whitney U-Test/Wilcoxon RankSum statistics:**  
1. Follow the instructions in the comments in UTest_Wilcoxon.py to choose any two algorithms from SEPS, PPOS and NESS and run ```python src/UTest_Wilcoxon.py``` to get the Test statistic and p-value (make sure to add the rewards output by the codes after running to get the reward values for yourself, current reward values provided are from the run on my system). 
