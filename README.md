## ContiNet: A neural network-based method for potential field continuation
   Program: ContiNet
   Code by: Guangxi Chen, Changli Yao, Sheng Zhang, Xianzhe Yin, Li Xiong
   Main address: School of Geophysics and Information Technology, China University of Geosciences, Beijing, China: 100083  
   E-mail: gxchen@email.cugb.edu.cn or clyao@cugb.edu.cn 
   The program folder contains the following parts:
## 1.Manual
   A PDF file introducing this Program, including all the information about ContiNet.
## 2.Environment.ym
   A Python environment file, within which the program can be run.
## 3.Code
   A folder containing code for ContiNet, including UpContiNet and DnContiNet.
## 4.Data
   Here we only provide some data related to testing and analysis. If users need to train from scratch, the complete training dataset can be obtained by contacting the author via email: gxchen@email.cugb.edu.cn. The data that consume less memory include:
   # (1) Data_1UTrainSet4_Part1, Data_1UTrainSet4_Part2
     These two data files contain some of the data from UTrainSet4.
   # (2) Data_3UTestSet
     This is the test data for Upward continuation.
   # (3) Data_4DTrainSet_Part
     This file contains some of the data from DTrainSet1 and DTrainSet2.
   # (4) Data_5DTestSet
     This is the test data for Downward continuation.
   # (5) Data_6Actualcase
     This folder contains the actual data used in this research.
## 5.Results
   A total of 7 output files are included, which contain the model training results of UC and DC, as well as the data analysis results. However, if the user needs to train from scratch to verify the reliability of the results, they need to download the complete dataset from aaa and perform relevant operations such as modifying the file path. The output result files include:
   # (1) Results_1DiscreteUC_1DataSet_Evaluation
     For Discrete UC, here, we analyzed the training results of different network layers and different datasets. However, for the sake of simplicity and due to memory constraints, we only present the prediction results for each dataset, as well as the model training results corresponding to UTrainSet4. The analysis results in this file correspond to Fig 6 in the manuscript.
   # (2) Results_1DiscreteUC_2Generalization
     Based on (1), We further analyzed the generalization ability of the model in UC height parameters, and provided the training and fine-tuned model parameters. The analysis results in this file correspond to Fig 7 in the manuscript.
   # (3) Results_1DiscreteUC_3FinalWeight
     In this document, we present the final weight training results for discrete UC. The analysis results in this file correspond to Fig 8 in the manuscript.
   # (4) Results_2TheoreticalUC_1SinglePoint
     For the theoretical UC, in this file, the model training parameters and historical records corresponding to a single UC calculation point, as well as the fitting weights of the least squares method, are provided.The analysis results in this file correspond to Fig 9 in the manuscript.
   # (5) Results_2TheoreticalUC_2AllPoints
     Similar to (4), this file corresponds to all the UC calculation points. The analysis results in this file correspond to Fig 10 and Fig 11 in the manuscript.
   # (6) Results_3Theoretical_DC
     This file contains all the results of the DC simulation. Including the model training results, the weight fitting results, as well as the analysis of training and testing samples.The analysis results in this file correspond to Fig 12, Fig 13, Fig 14, Fig 15, Fig 16, Fig 17, Fig 18 in the manuscript.
   # (7) Results_Actualcase
     This folder contains the actual data test results. The analysis results in this file correspond to Fig 20 in the manuscript.

# Please refer to each section for detailed information
