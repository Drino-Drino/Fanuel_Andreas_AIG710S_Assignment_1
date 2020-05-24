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
Pkg.add(PackageSpec(url = "https://github.com/JuliaPy/PyCall.jl.git"))
#+import PyCall
using PyCall

Pkg.add("StatsPlots")
using StatsPlots
Pkg.add(PackageSpec(url = "https://github.com/JuliaPy/PyPlot.jl")) 
using PyPlot

Pkg.add("ScikitLearn")
#Github link: Pkg.add(PackageSpec(url = "https://github.com/cstjean/ScikitLearn.jl.git")) 
using ScikitLearn

#=*************************************************DATA CLEANING***********************************************************
***************************************************DATA CLEANING**********************************************************=#

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
showall(Data)

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
#import libraries required for the next section
using ScikitLearn: fit!, predict, @sk_import, fit_transform! 
 @sk_import preprocessing: LabelEncoder 
 @sk_import model_selection: cross_val_score  
 @sk_import metrics: accuracy_score 
 @sk_import linear_model: LogisticRegression 
 @sk_import ensemble: RandomForestClassifier 
 @sk_import tree: DecisionTreeClassifier 
@sk_import preprocessing: LabelEncoder 

#ENCODING
labelencoder = LabelEncoder() 
categories = [2 3 4 5 6 7 8 9 10 20] #an array of the column numbers that need encoding 

for col in categories 
    Data[col] = fit_transform!(labelencoder, Data[col]) #performs encoding
end

#=Choose dependant and independant variables: all columns from 1 to 19 can have an impact on
whether or not the client decides to place a term deposit, so we use them all =#

#=Picking training and testing data at 80:20 ratio
80% of 41188 is 32950.4, so we reduce the training data to just below 80% i.e 32950 rows =#

XTrainData = convert(Matrix, Data[1:32950,1:19]) #Data training matrix for independant variables
XTestData = convert(Matrix, Data[32951:41188,1:19]) #Data Testing Matrix for independant variables
YtrainData = Data[1:32950,20] #Data training Vector for the dependant variable
YTestData = Data[32951:41188,20] #Data testing Vector for the dependant variable

#=****************************LOGISTIC REGRESSION, WITH L1 REGULARISATION IMPLEMENTATION*********************************************
******************************LOGISTIC REGRESSION, WITH L1 REGULARISATION IMPLEMENTATION********************************************=#

#Normalising the training design matrix
function scale_features(X)
    avg = mean(X, dims = 1)
    stdDev = std(X, dims=1)
    X_norm = (X .- avg) ./ stdDev

    return (X_norm, avg, stdDev);
end

#Normalising the tsting design matrix
function transform_features(X, avg, stdDev)
    X_norm = (X .-avg) ./stdDev
    return X_norm;
end

# Scale training features and get artificats for future use
X_train_scaled, avg, stdDev = scale_features(XTrainData);

# Transforming the testing features by using the learned artifacts
X_test_scaled = transform_features(XTestData, avg, stdDev);

#A function to apply the sigmoid activation to any supplied scalar/vector
function sigmoid(z)
    return 1 ./ (1 .+ exp.(.-z))    
end

#The regularised cost function computes the batch cost with a lambda penalty (λ) as well, using L1 regularisation
   
    function regularised_cost(X, y, θ, λ)
        m = length(y)
    
        # Sigmoid predictions at current batch
        h = sigmoid(X * θ)
    
        # left side of the cost function
        positive_class_cost = ((-y)' * log.(h))
    
        # right side of the cost function
        negative_class_cost = ((1 .- y)' * log.(1 .- h))
    
        # lambda effect
        lambda_regularization = (λ/(2*m) * sum(abs.(θ[2 : end])))
    
        # Current batch cost. Basically mean of the batch cost plus regularization penalty
        𝐉 = (1/m) * (positive_class_cost - negative_class_cost) + lambda_regularization
    
        # Gradients for all the theta members with regularization except the constant
        ∇𝐉 = (1/m) * (X') * (h-y) + (λ/m) # Penalise all members
    
        ∇𝐉[20] = (1/m) * (X[:, 20])' * (h-y) # Exclude the constant
    
        return (𝐉, ∇𝐉)
    end
 
    #=
    This function uses gradient descent to search for the weights that minimises the logit cost function.
        A tuple with learned weights vector (θ) and the cost vector (𝐉) 
        are returned.
        =#
        function logistic_regression_sgd(X, y, λ, fit_intercept=true, η=0.01, max_iter=1000)
            
            # Initialize some useful values
            m = length(y); # number of training examples
        
            if fit_intercept
                # Add a constant of 1s if fit_intercept is specified
                constant = ones(m, 1)
                X = hcat(constant, X)
            else
                X # Assume user added constants
            end
        
            # Use the number of features to initialise the theta θ vector
            n = size(X)[2]
            θ = zeros(n)
        
            # Initialise the cost vector based on the number of iterations
            𝐉 = zeros(max_iter)
        
            for iter in range(1, stop=max_iter)
        
                # Calcaluate the cost and gradient (∇𝐉) for each iter
                𝐉[iter], ∇𝐉 = regularised_cost(X, y, θ, λ)
        
                # Update θ using gradients (∇𝐉) for direction and (η) for the magnitude of steps in that direction
                θ = θ - (η * ∇𝐉)
            end
        
            return (θ, 𝐉)
        end

        # Using the  gradient descent to search for the optimal weights (θ)
θ, 𝐉 = logistic_regression_sgd(X_train_scaled, YtrainData, 0.0001, true, 0.3, 3000);

# Plot the cost vector
plot(𝐉, color="blue", title="Cost Per Iteration", legend=false,
     xlabel="Num of iterations", ylabel="Cost")

#=***********************************************************PREDICTIONS*****************************************************
*************************************************************PREDICTIONS***************************************************=#

#=This function uses the learned weights (θ) to make new predictions.
Predicted probabilities are returned=#
        
        function predict_proba(X, θ, fit_intercept=true)
            m = size(X)[1]
        
            if fit_intercept
                # Add a constant of 1s if fit_intercept is specified
                constant = ones(m, 1)
                X = hcat(constant, X)
            else
                X
            end
        
            h = sigmoid(X * θ)
            return h
        end
        
        
#=This function binarizes predicted probabilities using a threshold.
Default threshold is set to 0.5=#
        
        function predict_class(proba, threshold=0.5)
            return proba .>= threshold
        end
        
        
        # Training and validation score
        train_score = mean(YtrainData .== predict_class(predict_proba(X_train_scaled, θ)));
        test_score = mean(YTestData .== predict_class(predict_proba(X_test_scaled, θ)));
        
        # Training and validation score rounded to 4 decimals
        println("Training score: ", round(train_score, sigdigits=4))
        println("Testing score: ", round(test_score, sigdigits=4))

#=*************************************************************NOTES*******************************************************************
***************************************************************NOTES*******************************************************************
Commiting to git commands
git remote add origin <Link to GitHub Repo>     //maps the remote repo link to local git repo

git remote -v                                  //this is to verify the link to the remote repo 

git push -u origin master   
****************************************************************THE END*****************************************************************
****************************************************************THE END****************************************************************=#

