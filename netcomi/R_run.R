# Required packages
#install.packages("devtools")
#install.packages("BiocManager")
#For installing SpiecEasi,SPRING, NetCoMi (same steps)
#go to the folder in terminal
#(base) Rosas-MacBook-Pro:~ rosa$ cd /Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/
#git clone https://github.com/GraceYoon/SPRING.git
#(base) Rosas-MacBook-Pro:Code rosa$ tar czf SPRING.tar.gz SPRING
#In Rstudio Tools/Install packages from tar file
#rm(list=ls())
library(NetCoMi)
library(SpiecEasi)
library(SPRING)
#data("amgut1.filt")
#data1=read.csv("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/DATA/CountOTU.csv",header = TRUE,row.names = 1)
#dim(data1)
# zeroMethod =c("none","pseudo","multRepl","bayesMult")
# normMethod =c("TSS","CSS","COM","rarefy","mclr","VST")
# Rowsum=c()
# for (r in 1:length(normMethod)){
#   for (j in 1:length(zeroMethod)){
#       net_single3 <- netConstruct(data1, 
#                                  measure = "pearson",
#                                  zeroMethod = zeroMethod[j],
#                                  normMethod = normMethod[r], 
#                                  sparsMethod = "none", 
#                                  verbose = 3)
#     text=paste("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/DATA/Netcomi_data/",r,"_",j,".csv")
#     write.csv(net_single3$normCounts1, file = text)
#     Rowsum=rbind(Rowsum,rowSums(net_single3$normCounts1, na.rm = FALSE, dims = 1))
#   }
# }
####################################Normalizing Data
#############################################Phyrum
data1=read.csv("//Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/CountOTUY1_F_Phylum.csv",header = TRUE,row.names = 1)
dim(data1)
zeroMethod =c("none","pseudo","multRepl","bayesMult")
normMethod =c("TSS","CSS","COM","rarefy","mclr")
Rowsum=c()
for (r in 1:length(normMethod)){
  for (j in 1:length(zeroMethod)){
    net_single3 <- netConstruct(data1, 
                                measure = "pearson",
                                zeroMethod = zeroMethod[j],
                                normMethod = normMethod[r], 
                                sparsMethod = "none", 
                                verbose = 3)
    print(normMethod[r])
    print(zeroMethod[j])
    text=paste("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/Phylum/",r,"_",j,".csv")
    write.csv(net_single3$normCounts1, file = text)
    Rowsum=rbind(Rowsum,rowSums(net_single3$normCounts1, na.rm = FALSE, dims = 1))
  }
}
write.csv( Rowsum, file = '/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/Phylum/Rowsum.csv')
############################################Class
data1=read.csv("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/CountOTUY1_F_Class.csv",header = TRUE,row.names = 1)
dim(data1)
zeroMethod =c("none","pseudo","multRepl","bayesMult")
normMethod =c("TSS","CSS","COM","rarefy","mclr","VST")
Rowsum=c()
for (r in 1:length(normMethod)){
  for (j in 1:length(zeroMethod)){
    net_single3 <- netConstruct(data1, 
                                measure = "pearson",
                                zeroMethod = zeroMethod[j],
                                normMethod = normMethod[r], 
                                sparsMethod = "none", 
                                verbose = 3)
    print(r)
    print(j)
    text=paste("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/Class/",r,"_",j,".csv")
    write.csv(net_single3$normCounts1, file = text)
    Rowsum=rbind(Rowsum,rowSums(net_single3$normCounts1, na.rm = FALSE, dims = 1))
  }
}
write.csv( Rowsum, file = '/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/Class/Rowsum.csv')
############################################Family
data1=read.csv("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/CountOTUY1_F_Family.csv",header = TRUE,row.names = 1)
dim(data1)
zeroMethod =c("none","pseudo","multRepl","bayesMult")
normMethod =c("TSS","CSS","COM","rarefy","mclr","VST")
Rowsum=c()
for (r in 1:length(normMethod)){
  for (j in 1:length(zeroMethod)){
    net_single3 <- netConstruct(data1, 
                                measure = "pearson",
                                zeroMethod = zeroMethod[j],
                                normMethod = normMethod[r], 
                                sparsMethod = "none", 
                                verbose = 3)
    text=paste("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/Family/",r,"_",j,".csv")
    write.csv(net_single3$normCounts1, file = text)
    Rowsum=rbind(Rowsum,rowSums(net_single3$normCounts1, na.rm = FALSE, dims = 1))
  }
}
write.csv( Rowsum, file = '/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/Family/Rowsum.csv')
##########################################Order
data1=read.csv("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/CountOTUY1_F_Order.csv",header = TRUE,row.names = 1)
dim(data1)
zeroMethod =c("none","pseudo","multRepl","bayesMult")
normMethod =c("TSS","CSS","COM","rarefy","mclr","VST")
Rowsum=c()
for (r in 1:length(normMethod)){
  for (j in 1:length(zeroMethod)){
    net_single3 <- netConstruct(data1, 
                                measure = "pearson",
                                zeroMethod = zeroMethod[j],
                                normMethod = normMethod[r], 
                                sparsMethod = "none", 
                                verbose = 3)
    text=paste("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/Order/",r,"_",j,".csv")
    write.csv(net_single3$normCounts1, file = text)
    Rowsum=rbind(Rowsum,rowSums(net_single3$normCounts1, na.rm = FALSE, dims = 1))
  }
}
write.csv( Rowsum, file = '/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/Order/Rowsum.csv')
##########################################Genus
data1=read.csv("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/CountOTUY1_F_Genus.csv",header = TRUE,row.names = 1)
dim(data1)
zeroMethod =c("none","pseudo","multRepl","bayesMult")
normMethod =c("TSS","CSS","COM","rarefy","mclr","VST")
Rowsum=c()
for (r in 1:length(normMethod)){
  for (j in 1:length(zeroMethod)){
    net_single3 <- netConstruct(data1, 
                                measure = "pearson",
                                zeroMethod = zeroMethod[j],
                                normMethod = normMethod[r], 
                                sparsMethod = "none", 
                                verbose = 3)
    text=paste("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/Genus/",r,"_",j,".csv")
    write.csv(net_single3$normCounts1, file = text)
    Rowsum=rbind(Rowsum,rowSums(net_single3$normCounts1, na.rm = FALSE, dims = 1))
  }
}
write.csv( Rowsum, file = '/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/Genus/Rowsum.csv')

########################################################
########################################################
#############################################Phyrum
data1=read.csv("//Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/CountOTUY1_F_Phylum.csv",header = TRUE,row.names = 1)

##########################################Order
data1=read.csv("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/CountOTUY1_F_Order.csv",header = TRUE,row.names = 1)

############################################Class
data1=read.csv("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/CountOTUY1_F_Class.csv",header = TRUE,row.names = 1)

############################################Family
data1=read.csv("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/CountOTUY1_F_Family.csv",header = TRUE,row.names = 1)

##########################################Genus
data1=read.csv("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/CountOTUY1_F_Genus.csv",header = TRUE,row.names = 1)
#############################################
#############################################
############################################
net_single <- netConstruct(data1,
                           filtTax = "highestFreq",
                           filtTaxPar = list(highestFreq = 200),
                           #filtSamp = "totalReads",
                           #filtSampPar = list(totalReads = 1000),
                           measure = "spring",
                           measurePar = list(nlambda=10, 
                                             rep.num=10),
                           normMethod = "none", 
                           zeroMethod = "none",
                           sparsMethod = "none", 
                           dissFunc = "signed",
                           verbose = 3,
                           seed = 123456)

props_single <- netAnalyze(net_single, 
                           centrLCC = TRUE,
                           clustMethod = "cluster_fast_greedy",
                           hubPar = "eigenvector",
                           weightDeg = FALSE, normDeg = FALSE)

props_single <- netAnalyze(net_single, 
                           centrLCC = TRUE,
                           clustMethod = "cluster_fast_greedy",
                           hubPar = "eigenvector",
                           weightDeg = FALSE, normDeg = FALSE)

s<-summary(props_single, numbNodes = 5L)

capture.output(s, file ="/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Report/summary_Family.txt")
tiff(filename="/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Report/Netspring_Family.tiff", width=2300, height=2000, res=300)

p<-plot(props_single, 
        nodeColor = "cluster", 
        nodeSize = "eigenvector",
        title1 = "Network on OTU level with SPRING associations", 
        showTitle = TRUE,
        cexTitle = 1)
legend(0.5, 0.9, cex = 1, title = "estimated association:",
       legend = c("+","-"), lty = 1, lwd = 3, col = c("#009900","red"), 
       bty = "n", horiz = TRUE)
dev.off()



# Let’s improve the visualization by changing the following arguments:
#   
# repulsion = 0.8: Place the nodes further apart
# rmSingles = TRUE: Single nodes are removed
# labelScale = FALSE and cexLabels = 1.6: All labels have equal size and are enlarged to improve readability of small node’s labels
# nodeSizeSpread = 3 (default is 4): Node sizes are more similar if the value is decreased. This argument (in combination with cexNodes) is useful to enlarge small nodes while keeping the size of big nodes.


###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
######################Costructing based on correlation methods

net_single2 <- netConstruct(data1,  
                            measure = "pearson",
                            normMethod = "clr", 
                            zeroMethod = "multRepl",
                            sparsMethod = "threshold", 
                            thresh = 0.3,
                            verbose = 3)

props_single2 <- netAnalyze(net_single2, clustMethod = "cluster_fast_greedy")

plot(props_single2, 
     nodeColor = "cluster", 
     rmSingles = TRUE,
     repulsion = 0.8,
     nodeSize = "eigenvector",
     title1 = "Network on OTU level with Pearson correlations", 
     showTitle = TRUE,
     cexTitle = 2.3)

legend(0.7, 1.1, cex = 2.2, title = "estimated correlation:", 
       legend = c("+","-"), lty = 1, lwd = 3, col = c("#009900","red"), 
       bty = "n", horiz = TRUE)

#############################################################
############################################################
###Comparing Networks in zero and one label

net_season <- netConstruct(data = data1[1:100,], 
                           data2 = data1[101:200,],  
                           filtTax = "highestVar",
                           filtTaxPar = list(highestVar = 40),
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

sm<-summary(props_season, numbNodes = 40L)

plot(props_season, 
     sameLayout = FALSE, 
     nodeColor = "cluster",
     #nodeSize = "degree",
     nodeSize = "mclr",
     rmSingles = TRUE,
     cexNodes = 1.5, 
     cexLabels = 2.5,
     cexHubLabels = 3,
     cexTitle = 3.7,
     groupNames = c("label 0", "label 1"),
     hubBorderCol  = "gray40")



###################################Comparing Networkd difference measures

comp_season <- netCompare(props_season, permTest = FALSE, verbose = FALSE)

diff<-summary(comp_season, 
        groupNames = c("label 0", "label 1"),
        showCentr = c("degree", "between", "closeness"), 
        numbNodes = 30)

top_Deg=diff$topProps$topDeg
top_Betw=diff$topProps$topBetw
top_Close=diff$topProps$topClose

####3return All top 
# m1=merge(top_Deg,top_Betw,by="row.names",all.x=TRUE)
# row.names(m1)<-m1[,1]
# m2=merge(m1,top_Close,by="row.names",all.x=TRUE)

####3return intersection of them
m1=merge(top_Deg,top_Betw,by="row.names")
row.names(m1)<-m1[,1]
m2=merge(m1,top_Close,by="row.names")
m2=m2[,2:dim(m2)[2]]
rownames(m2) <- m2[,1]
m2[,1] <- NULL
names(m2) <- make.names(names(m2), unique=TRUE)
capture.output(m2, file ='/Users/rosa/Desktop/feature_diff_label0_1.csv')



