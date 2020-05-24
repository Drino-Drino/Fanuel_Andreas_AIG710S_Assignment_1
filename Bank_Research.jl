#importing required libraries
import Pkg
using Pkg
using CSV
using Statistics
using Plots
using DataFrames
Pkg.add("StatsBase")
#Github link: Pkg.add(PackageSpec(url = "https://github.com/JuliaStats/StatsBase.jl.git"))
using StatsBase
Pkg.add("ScikitLearn")
#Github link: Pkg.add(PackageSpec(url = "https://github.com/cstjean/ScikitLearn.jl.git"))
using ScikitLearn
Pkg.build("PyCall")
Pkg.build("PyPlot")



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



StatsBase.mode(Data[:job])
#since the mode is admin, we replace all unknown values with admin
Data[Data[:job].=="unknown",:job] = StatsBase.mode(Data[:job])
#Checking the Categories (different values) of marital
Pkg.add("CategoricalArrays")
using CategoricalArrays
MaritalVector = CategoricalArray(Data[:marital]) #A vector of the marital column
levels(MaritalVector) #Checking the different Categories in the marital column, incase there are missing values

#We replace unknown by the mode of the marital column
StatsBase.mode(Data[:marital]) #checking the mode
Data[Data[:marital].=="unknown",:marital] = StatsBase.mode(Data[:marital])

#We repeat the above process for all other data columns
EducationVector = CategoricalArray(Data[:education])
levels(EducationVector) #checking the categories, we notice "unknown"
StatsBase.mode(Data[:education]) #checking the mode != unknown
Data[Data[:education].=="unknown",:education] = StatsBase.mode(Data[:education])

#default column cleaning
DefaultVector = CategoricalArray(Data[:default]) #we notice "unknown" values straight away
levels(DefaultVector) #check for unknown values
StatsBase.mode(Data[:default]) #we check the mode and it is = no
Data[Data[:default].=="unknown",:default] = StatsBase.mode(Data[:default])

#housing column cleaning
HousingVector = CategoricalArray(Data[:housing])
levels(HousingVector)
StatsBase.mode(Data[:housing])
Data[Data[:housing].=="unknown",:housing] = StatsBase.mode(Data[:housing])

#Cleaning loan column
LoanVector = CategoricalArray(Data[:loan])
levels(LoanVector)
StatsBase.mode(Data[:loan])
Data[Data[:loan].=="unknown",:loan] = StatsBase.mode(Data[:loan])

#Cleaning contact column
ContactVector = CategoricalArray(Data[:contact])
levels(ContactVector) #no unknown/empty values

#Cleaning The Month Column
MonthVector = CategoricalArray(Data[:month])
Categories = levels(MonthVector) #there are 10 categories that are not fully displayed in console
#we create a for loop to display all the categories
for i in 1:10 #Julia arrays count from 1
    println(
        Categories[i])
end  #The loop returned month values only, so no editing required here

#Cleaning Day of week
DOWVector = CategoricalArray(Data[:day_of_week])
levels(DOWVector) #returned days only

#Cleaning Duration Column
DurationVector = CategoricalArray(Data[:duration])

Categories = levels(DurationVector) #has numerous(1544) different values,
for i in 1:1544 #Iterate through all values and print them
    print(
        Categories[i])
end  #returned no missing values

#Cleaning Campaign Column
CampaignVector = CategoricalArray(Data[:campaign])
Categories = levels(CampaignVector) #has 42 categories
for i in 1:42 #Iterate through all values and print them
    print(
        Categories[i])
end  #returned no missing values

#Cleaning Pdays Column //Repeat above code
PdaysVector = CategoricalArray(Data[:pdays])
Categories = levels(PdaysVector)
for i in 1:27 #Iterate through all values and print them
    print(
        Categories[i])
end  #returned no missing values

#Cleaning Previous Column
PreviousVector = CategoricalArray(Data[:previous])
Categories = levels(PreviousVector)
for i in 1:8 #Iterate through all values and print them
    print(
        Categories[i])
end  #returned no missing values

#Cleaning Outcome column
POutcomeVector = CategoricalArray(Data[:poutcome])
Categories = levels(POutcomeVector) #3 outcomes of which one is "non-existants"
StatsBase.mode(Data[:poutcome])#we look for the mode, and the result is mode is "nonexistent"
#since the majority of the data of the column poutcome is unavailable, we delete it!!
select!(Data, Not(:poutcome))

#cleaning emp.var.rate
EmpVarRateVector = CategoricalArray(Data[:15])
Categories =levels(EmpVarRateVector) #returned 10 categories with no missing missing/ambigouos data


#Cleaning cons.price.idx 
ConsPriceVector = CategoricalArray(Data[:16])
Categories = levels(ConsPriceVector)#returns a 26 element arrays
#print all the categories
for i in 1:26 #Iterate through all values and print them
    print(
        Categories[i])
end #returns actual values only (no missing data)

#Cleaning cons.conf.idx
ConsConfVector = CategoricalArray(Data[:17])
Categories = levels(ConsConfVector)#returns a 26 element arrays
#print all the categories
for i in 1:26 #Iterate through all values and print them
    print(
        Categories[i])
end #returns actual values only (no missing data)

#Cleaning euribor3m
ConsEuriVector = CategoricalArray(Data[:euribor3m]) 
Categories = levels(ConsEuriVector)#returns a 316 element arrays 
#print all the categories
for i in 1:316 #Iterate through all values and print them
    print(
        Categories[i])
end #returns actual values only (no missing data)

#Cleaning nr.employed
NrEmployedVector = CategoricalArray(Data[:19])
Categories = levels(NrEmployedVector)#returns a 11 element arrays, all data displayed with no missing values

#Cleaning y
YVector = CategoricalArray(Data[:y])
Categories = levels(YVector)#returns a 2 element arrays, yes or no being the only outcomes, therefore no edit required

#Label Encoding all non numerical columns to numbers
@sk_import preprocessing: LabelEncoder #=
labelencoder = LabelEncoder() 
categories = [2 3 4 5 6 7 8 9 10 20] #Column index numbers of categorical data. *********** last contact duration is taken as numeric despite 
                                #the metadata description ****** number of columns reduced to 20 due to the removal of the poutcome column

for col in categories 
    Data[col] = fit_transform!(labelencoder, Data[col]) 
end


    for name in names(Data)              *******Code for printing all column names *********
       print(":", name, ", ")
      end
  
************** add a package from a link  *****************

#train[isna.(train[:Married]), :Married] = mode(dropna(train[:Married])) 
#replace 0.0 of loan amount with the mean of loan amount 
#train[train[:LoanAmount] .== 0, :LoanAmount] = floor(mean(dropna(train[:LoanAmount])))=#


#DataMatrix = convert(Matrix, Data)

#=Commiting to git commands
git remote add origin <Link to GitHub Repo>     //maps the remote repo link to local git repo

git remote -v                                  //this is to verify the link to the remote repo 

git push -u origin master   =#