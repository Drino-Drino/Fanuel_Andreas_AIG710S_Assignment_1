#importing required libraries
using CSV
using Statistics
using Plots
using DataFrames

#import the data file, while also allowing for the data to be edited
Data = CSV.read("C:\\Users\\andre\\Documents\\Nust\\AIG\\Ass1_Project\\bank-additional-full.csv", copycols = true);
DataMatrix = convert(Matrix, Data)

#Data Exploration
#Display Column Names
names(Data)

#Size of spreadsheet
size(Data)

#Display all columns and first 3 rows, to see Data types of the different columns
head(Data, 3)

#=Checking for columns with missing values below shows that all cells are filled (no missing values)
    the function describes all the columns by giving name, mean, min, median, max,missing values=#
describe(Data)

#Display entire table
#showall(Data)

#=From the result above, we realise that there are missing values that are marked "unknown" or "nonexistent"
we therefore need to replace these values with mean (for numerical data) and mode for categorical data=#

#We have every participant's age, so we will not change anything in the column

#We need to replace unknown in job with the mode of job i.e admin

import Pkg; 
#=run this coder in terminal to add StatsBase:
pkg> add https://github.com/JuliaStats/StatsBase.jl.git =#
using StatsBase
StatsBase.mode(Data[:job])
#since the mode is admin, we replace all unknown values with admin
Data[Data[:job].=="unknown",:job] = StatsBase.mode(Data[:job])


#Checking the Categories (different values) of marital
Pkg.add("CategoricalArrays")
using CategoricalArrays
MaritalVector = CategoricalArray(Data[:marital]) #A vector of the marital column
levels(BVector) #Checking the different Categories in the marital column, incase there are missing values

#We replace unknown by the mode of the marital column
StatsBase.mode(Data[:marital]) #checking the mode
Data[Data[:marital].=="unknown",:marital] = StatsBase.mode(Data[:marital])

#We repeat the above process for all categorical data columns


#train[isna.(train[:Married]), :Married] = mode(dropna(train[:Married])) 
#replace 0.0 of loan amount with the mean of loan amount 
#train[train[:LoanAmount] .== 0, :LoanAmount] = floor(mean(dropna(train[:LoanAmount])))


#DataMatrix = convert(Matrix, Data)