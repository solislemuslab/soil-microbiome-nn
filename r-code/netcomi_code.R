##########################################################################################Packages
library(NetCoMi)
library(SpiecEasi)
library(SPRING)
###########################################################################################PART1:Normalization
####################################Normalizing OTU Data
start_time <- Sys.time()
unlink("otu/Icon\r")
path_otu="otu"
file_list_otu=list.files(path_otu)

zeroMethod =c("none","pseudo","multRepl","bayesMult")
normMethod =c("TSS","CSS","COM","rarefy","mclr")
for (j in 1:length(file_list_otu)){
  
  file_list_otu_file=list.files(file.path(path_otu, file_list_otu[j]),pattern="^Count")
  data1=read.csv(file.path(path_otu, file_list_otu[j],file_list_otu_file),header = TRUE,row.names = 1)
  print(dim(data1))
  for (r in 1:length(normMethod)){
    for (k in 1:length(zeroMethod)){
      net_single3 <- netConstruct(data1, 
                                  measure = "pearson",
                                  zeroMethod = zeroMethod[k],
                                  normMethod = normMethod[r], 
                                  sparsMethod = "none", 
                                  verbose = 1)
      write.csv(net_single3$normCounts1,paste(file.path(path_otu,file_list_otu[j],sep=""),r,'_',k,'.csv'))
    }
  }
}
end_time <- Sys.time()
end_time - start_time
###########################################################################################Part2: Normalized Augmented data
start_time <- Sys.time()
unlink("augumented_otu/Icon\r")
path_otu="augumented_otu"
file_list_otu=list.files(path_otu)

zeroMethod =c("none","pseudo","multRepl","bayesMult")
normMethod =c("TSS","CSS","COM","rarefy","mclr")
for (j in 1:length(file_list_otu)){
  
  file_list_otu_file=list.files(file.path(path_otu, file_list_otu[j]),pattern="^Count")
  data1=read.csv(file.path(path_otu, file_list_otu[j],file_list_otu_file),header = TRUE,row.names = 1)
  print(dim(data1))
  for (r in 1:length(normMethod)){
    for (k in 1:length(zeroMethod)){
      net_single3 <- netConstruct(data1, 
                                  measure = "pearson",
                                  zeroMethod = zeroMethod[k],
                                  normMethod = normMethod[r], 
                                  sparsMethod = "none", 
                                  verbose = 1)
      write.csv(net_single3$normCounts1,paste(file.path(path_otu,file_list_otu[j],sep=""),r,'_',k,'.csv'))
    }
  }
}
end_time <- Sys.time()
end_time - start_time
#############################################################################################PART3:Constructing Network
start_time <- Sys.time()
unlink("otu_count/Icon\r")
unlink("response_netcomi/Icon\r")
path_otu="otu_count/"
path_response="response_netcomi/"
file_list_otu=list.files(path_otu, pattern="*.csv")
file_list_response=list.files(path_response)
for (i in 1:length(file_list_response)){
  response_file=list.files(file.path(path_response,file_list_response[i]),pattern="^response")
  response=read.csv(file.path(path_response, file_list_response[i],response_file),row.names = 1)
  for (j in 1:length(file_list_otu)){
    data1=read.csv(file.path(path_otu, file_list_otu[j]),header = TRUE,row.names = 1)
    data=merge(data1,response,by="row.names",row.names = 1)
    row.names(data)=data[,1]
    data[,1]<-NULL
    d1=data[data[,dim(data)[2]]==0,]
    d2=data[data[,dim(data)[2]]==1,]
    d1['x1']<-NULL
    d2['x1']<-NULL
    print(dim(data))
    print(dim(d1))
    print(dim(d2))
###Comparing Networks in zero and one label
    net_season <- netConstruct(data = d1, 
                               data2 = d2,  
                               filtTax = "highestVar",
                               filtTaxPar = list(highestVar = dim(data)[2]),
                               measure = "spring",
                               measurePar = list(nlambda=10, 
                                                 rep.num=10),
                               normMethod = "none", 
                               zeroMethod = "none",
                               sparsMethod = "none", 
                               dissFunc = "signed",
                               verbose = 3,
                               seed = 123456)
    
    props_season <- netAnalyze(net_season, 
                               centrLCC = FALSE,
                               avDissIgnoreInf = TRUE,
                               sPathNorm = FALSE,
                               clustMethod = "cluster_fast_greedy",
                               hubPar = c("degree", "between", "closeness","eigenvector"),
                               hubQuant = 0.1,
                               lnormFit = TRUE,
                               normDeg = FALSE,
                               normBetw = FALSE,
                               normClose = FALSE,
                               normEigen = FALSE)

###################################Comparing Network difference measures
     comp_season <- netCompare(props_season, permTest = FALSE, verbose = FALSE)
    
    diff<-summary(comp_season, 
                  groupNames = c("label0", "label1"),
                  showCentr = c("degree", "between", "closeness"), 
                  numbNodes = round(dim(data)[2]))
    
    top_Deg=diff$topProps$topDeg
    top_Betw=diff$topProps$topBetw
    top_Close=diff$topProps$topClose
    
    ####3return intersection of them
    m1=merge(top_Deg,top_Betw,by="row.names")
    row.names(m1)<-m1[,1]
    m2=merge(m1,top_Close,by="row.names")
    m2=m2[,2:dim(m2)[2]]
    rownames(m2) <- m2[,1]
    m2[,1] <- NULL
    names(m2) <- make.names(names(m2), unique=TRUE)
    write.csv(data.frame(m2),file =file.path(path_response, file_list_response[i],file_list_otu[j]),row.names = TRUE)
  }
}
end_time <- Sys.time()
end_time - start_time
####################################################################################





