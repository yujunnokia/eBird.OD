# TODO: generate synthetic data from the OD model
#
# Author: Jun Yu
# Version: Jan 16, 2012
###############################################################################

source("OD.R")

#
# generate synthetic data from OD model
#
# Args:
#   nSites: number of sites
#	visits: a vector recording number of visits per site
#	alpha: occupancy parameter
#	beta: detection parameter
#	FPR: false positive probability (rate) is \in [0,1] with default value being 0
#
# Returns:
#   synthetic data generated from the model including detection history, occupancy 
#	covaraites, detection covariates and true occupancy status.
#
GenerateData <- function(nSites,visits,alpha,beta,FPR=0) 
{
    nOccCovs <- length(alpha)
    nDetCovs <- length(beta)
    nVisits  <- max(visits)
    
    # generate occupancy and detection covariates
    occCovs <- rnorm(nSites * nOccCovs)
    dim(occCovs) <- c(nSites, nOccCovs)
    occCovs[,1] <- array(1, c(nSites, 1))
    detCovs <- rnorm(nSites * nVisits * nDetCovs)
    dim(detCovs) <- c(nSites, nVisits, nDetCovs)
    detCovs[,,1] <- array(1, c(nSites, nVisits, 1))
    
    trueOccs <- array(0, c(nSites, 1))
    trueOccs <- runif(nSites) < Logistic(occCovs %*% alpha)
    
    detHists <- array(0, c(nSites, nVisits))
    FPs <- array(0, c(nSites, nVisits))
    for (i in 1:nSites) {
        for (t in 1:visits[i]) {
            isDetected <- runif(1) < Logistic(detCovs[i,t,] %*% beta)
            # false positives 
            isFalsePositive <- runif(1) < FPR
            if (isFalsePositive == 1) {
                if (isDetected == 1) {
                    detHists[i,t] <- 1
                }
                if (isDetected == 1 && trueOccs[i] == 0) {
                    FPs[i,t] <- 1
                }
            } else {
                if (trueOccs[i] == 1 && isDetected == 1) {
                    detHists[i,t] <- 1
                }				
            }
        } # t
    } # i
    
    retData <- list(detHists=detHists, occCovs=occCovs, detCovs=detCovs, trueOccs=trueOccs, FPs=FPs)
    return(retData)
}
