# TODO: Learn the model parameter from the synthetic data 
#
# Author: Jun Yu
# Version: Jan 16, 2012
##################################################################

rm(list=ls())

#setwd("C:/Jun_home/workspace/eBird.OD")
#setwd("/Users/yujunnokia/Documents/workspace/eBird.OD")
setwd("/Users/yujunnokia/workspace/eBird.OD")
source("OD.R")
source("OD.synthData.R")

library("lattice")
library("Matrix")
library("glmnet")

######################
# experiment settings
######################
nTrSites <- 500  # number of training sites
nTeSites <- 500  # number of testing sites
nVisits <- 3  # number of visits to each site
nOccCovs <- 5  # number of occupancy covariates
nDetCovs <- 5  # number of detection covariates
nParams <- nOccCovs + nDetCovs + 2  # total number of paramters. 2 is the two intercept terms from occupancy and detection parts
nRandomRestarts <- 1  # number of random restarts  
falsePositiveRate <- 0 # proportion of false positives added into the data

#################
# regularization
#################
regType <- 2 # regularization types: 0 for none, 1 for L1, 2 for L2
lambda <- lambdaO <- lambdaD <- 0.01  # regularization paramters

#######################
# set model parameters
#######################
alpha <- rnorm(nOccCovs)*3  # random sample alpha
beta <- rnorm(nDetCovs)*3  # random sample beta
intercept <- 1  # define intercept
alpha <- c(intercept, alpha)  # add intercept term to alpha
beta <- c(intercept, beta)  # add intercept term to beta
trueParams <- list()
trueParams[["alpha"]] <- alpha
trueParams[["beta"]]  <- beta

########################
# generate testing data
########################
teVisits <- array(nVisits, nTeSites)
teData <- GenerateData(nTeSites, teVisits, trueParams, falsePositiveRate)
teDetHists <- teData$detHists
teOccCovs <- teData$occCovs
teDetCovs <- teData$detCovs
teTrueOccs <- teData$trueOccs

#########################
# generate training data
#########################
trVisits <- array(0, c(nTrSites,1))
for (i in 1:nTrSites) {
    isMultipleVisits <- runif(1) < 0.5
    if (isMultipleVisits == TRUE) {
        trVisits[i] <- round(runif(1, min=2, max=nVisits))
    } else {
        trVisits[i] <- 1
    }
} # i
trData <- GenerateData(nTrSites, trVisits, trueParams, falsePositiveRate)
trDetHists <- trData$detHists
trOccCovs <- trData$occCovs
trDetCovs <- trData$detCovs
trTrueOccs <- trData$trueOccs

#####################
# get Bayes rates
#####################
{
    # get occupancy rate and detection rate
    teOccProb <- array(0, c(nTeSites, 1))
    teDetProb <- array(0, c(nTeSites, nVisits))
    predDetHists <- array(0, c(nTeSites, nVisits))
    for (i in 1:nTeSites) {
        teOccProb[i] <- Logistic(teOccCovs[i,] %*% alpha)
        for (t in 1:teVisits[i]) {
            teDetProb[i,t] <- Logistic(teDetCovs[i,t,] %*% beta)
        } # t
    } # i
    for (t in 1:nVisits) {
        predDetHists[,t] <- round(teOccProb) * round(teDetProb[,t])
    } # t
    bayesOcc <- sum(round(teOccProb) == teTrueOccs) / nTeSites
    bayesDet <- sum(sum(predDetHists == teDetHists)) / (sum(teVisits))
#    cat("bayes occupancy rate is ", bayesOcc, "\n")
#    cat("bayes detection rate is ", bayesDet, "\n")
}

###########
# OD model
###########
{
    # run OD
    params <- RandomRestartEM(trDetHists,trOccCovs,trDetCovs,trVisits,regType,lambdaO,lambdaD,nRandomRestarts)
    alphaOD <- params$alpha
    betaOD <- params$beta
    
    # compute occupancy rate and detection rate
    teOccProb <- array(0,c(nTeSites,1))
    teDetProb <- array(0,c(nTeSites,nVisits))
    predDetHists <- array(0,c(nTeSites,nVisits))
    for (i in 1:nTeSites) {
        teOccProb[i] <- Logistic(teOccCovs[i,] %*% alphaOD)
        for (t in 1:teVisits[i]) {
            teDetProb[i,t] <- Logistic(teDetCovs[i,t,] %*% betaOD)
        }
    }
    for (t in 1:nVisits) {
        predDetHists[,t] <- round(teOccProb)*round(teDetProb[,t])
    }
    modelOcc <- sum(round(teOccProb) == teTrueOccs) / nTeSites
    modelDet <- sum(sum(predDetHists == teDetHists)) / (sum(teVisits))
    cat("------------------------------\n")
    cat("bayes occupancy rate is ", bayesOcc, "\n")
    cat("model occupancy rate is ", modelOcc, "\n")
    cat("bayes detection rate is ", bayesDet, "\n")
    cat("model detection rate is ", modelDet, "\n")
    
    
    # predict Z on test data
    teBayesOccProb <- array(0,c(nTeSites,1))
    teModelOccProb <- array(0,c(nTeSites,1))
    for (i in 1:nTeSites) {
        teBayesOccProb[i] <- PredictOcc(trueParams,teOccCovs[i,],teDetCovs[i,,],teDetHists[i,],teVisits[i]) 
        teModelOccProb[i] <- PredictOcc(params,teOccCovs[i,],teDetCovs[i,,],teDetHists[i,],teVisits[i]) 
    } # i
    trueOcc <- sum(round(teBayesOccProb) == teTrueOccs) / nTeSites
    predOcc <- sum(round(teModelOccProb) == teTrueOccs) / nTeSites
    cat("------------------------------\n")
    cat("True occupancy prediction is ",trueOcc,"\n")
    cat("Model occupancy prediction is ",predOcc,"\n")
    
    # predict Y on test data
    teBayesDetHists <- array(0,c(nTeSites,nVisits))
    teModelDetHists <- array(0,c(nTeSites,nVisits))
    for (i in 1:nTeSites) {
        for (t in 1:teVisits[i]) {
            teBayesDetHists[i,t] <- PredictDet(trueParams,teOccCovs[i,],teDetCovs[i,t,]) 
            teModelDetHists[i,t] <- PredictDet(params,teOccCovs[i,],teDetCovs[i,t,]) 
        }
    }
    trueDet <- sum(sum(round(teBayesDetHists) == teDetHists)) / (sum(teVisits))
    predDet <- sum(sum(round(teModelDetHists) == teDetHists)) / (sum(teVisits))
    cat("True detection prediction is ",trueDet,"\n")
    cat("Model detection prediction is ",predDet,"\n")
    
    # compute MSE
    MSE <- sum((alpha - alphaOD)^2 + (beta - betaOD)^2) / nParams
    cat("------------------------------\n")
    cat("MSE is ",MSE,"\n")
}

#################################################
# run logistic regression to predict detection Y
#################################################
{
    trOccDetCovs <- NULL 
    trDetections <- NULL 
    for (i in 1:nTrSites) {
        for (t in 1:trVisits[i]) {
            trOccDetCovs <- rbind(trOccDetCovs,c(trOccCovs[i,2:(nOccCovs+1)],trDetCovs[i,t,2:(nDetCovs+1)]))
            trDetections <- rbind(trDetections,trDetHists[i,t])
        }
    }
    
    # train GLMNET with L2 norm
    LRM <- glmnet(trOccDetCovs,trDetections,family="binomial",alpha=0)
    
    # get occupancy rate
    teOccDetCovs <- NULL 
    teDetections <- NULL 
    for (i in 1:nTeSites) {
        for (t in 1:teVisits[i]) {
            teOccDetCovs <- rbind(teOccDetCovs,c(teOccCovs[i,2:(nOccCovs+1)],teDetCovs[i,t,2:(nDetCovs+1)]))
            teDetections <- rbind(teDetections,teDetHists[i,t])
        }
    }
    
    # predict
    predDetHists <- predict(LRM,type="response",newx=teOccDetCovs,s=lambda)
    predDetHists[predDetHists >= 0.5] <- 1
    predDetHists[predDetHists < 0.5] <- 0
    
    LRMDet <- sum(predDetHists == teDetections) / sum(teVisits)
    cat("LRM detection rate is ",LRMDet,"\n")
}

