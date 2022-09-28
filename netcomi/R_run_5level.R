#rm(list=ls())
library(NetCoMi)
library(SpiecEasi)
library(SPRING)
#########################################################################################
#########################################################################################
###################################feature_selection
#path_otu="/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/python code local/Main Data Files/filter_data_na"
unlink("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/python code local/Main Data Files/filter_data_na2/Icon\r")
unlink("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/python code local/Main Data Files/response_netcomi/Icon\r")
path_otu="/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/python code local/Main Data Files/filter_data_na2/"
path_response="/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/python code local/Main Data Files/response_netcomi"
file_list_otu=list.files(path_otu, pattern="*.csv")
file_list_response=list.files(path_response)
for (i in 1:length(file_list_response)){
        response_file=list.files(file.path(path_response,file_list_response[i]),pattern="^response")
        response=read.csv(file.path(path_response, file_list_response[i],response_file),row.names = 1)
        response[,1:2]<-NULL
        for (j in 1:length(file_list_otu)){
                data1=read.csv(file.path(path_otu, file_list_otu[j]),header = TRUE,row.names = 1)
                data=merge(data1,response,by="row.names",row.names = 1)
                row.names(data)=data[,1]
                data[,1]<-NULL
                d1=data[data[,dim(data)[2]]==0,]
                d2=data[data[,dim(data)[2]]==1,]
                colnames(data)[dim(data)[2]]<-"y_binary"
                d1['y_binary']<-NULL
                d2['y_binary']<-NULL
                print(dim(data))
                print(dim(d1))
                print(dim(d2))
                ###Comparing Networks in zero and one label
                net_season <- netConstruct(data = d1, 
                                           data2 = d2,  
                                           filtTax = "highestVar",
                                           filtTaxPar = list(highestVar = min(dim(data)[2],200)),
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
                tiff(file.path(path_response, file_list_response[i],"Fig.tiff"), width=2300, height=2000, res=300)
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
                     hubBorderCol  = "gray40")
                dev.off()
                
                
                
                
                
                ###################################Comparing Networkd difference measures
                
                comp_season <- netCompare(props_season, permTest = FALSE, verbose = FALSE)
                
                diff<-summary(comp_season, 
                              groupNames = c("label 0", "label 1"),
                              showCentr = c("degree", "between", "closeness"), 
                              numbNodes = min(100,round(dim(data)[2])))
                
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
                capture.output(m2, file =file.path(path_response, file_list_response[i],file_list_otu[j]))
                #capture.output(m2, file ='/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Result-total/feature-selection-netcomi/feature-selection-netcomi-no_tuber_scab/feature_diff_label0_1_Phylum.csv')
        
        }
}



#########################################################################################
#########################################################################################
####################################Normalizing Data
unlink("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/python code local/Main Data Files/filter_data_na2/Icon\r")
path_otu="/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/python code local/Main Data Files/filter_data_na2"
file_list_otu=list.files(path_otu)

#file_generated=list.files(path_otu)[grep("*.csv",All,invert=TRUE)]
zeroMethod =c("none","pseudo","multRepl","bayesMult")
normMethod =c("TSS","CSS","COM","rarefy","mclr")
for (j in 1:length(file_list_otu)){
        
        file_list_otu_file=list.files(file.path(path_otu, file_list_otu[j]),pattern="^Count")
        data1=read.csv(file.path(path_otu, file_list_otu[j],file_list_otu_file),header = TRUE,row.names = 1)
        print(dim(data1))
        Rowsum=c()
        for (r in 1:length(normMethod)){
                for (k in 1:length(zeroMethod)){
                        net_single3 <- netConstruct(data1, 
                                            measure = "pearson",
                                            zeroMethod = zeroMethod[k],
                                            normMethod = normMethod[r], 
                                            sparsMethod = "none", 
                                            verbose = 1)
                write.csv(net_single3$normCounts1,paste(file.path(path_otu,file_list_otu[j],sep=""),r,'_',k,'.csv'))
                Rowsum=rbind(Rowsum,rowSums(net_single3$normCounts1, na.rm = FALSE, dims = 1))
                }
        }
write.csv( Rowsum, paste(file.path(path_otu,file_list_otu[j],sep=""),"Rowsum.csv"))
}


#########################################################################################
#########################################################################################PLOT
###################################feature_selection
#path_otu="/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/python code local/Main Data Files/filter_data_na"
unlink("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/python code local/Main Data Files/filter_data/Icon\r")
unlink("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/python code local/Main Data Files/response_netcomi/Icon\r")
path_otu="/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/python code local/Main Data Files/filter_data/"
path_response="/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/python code local/Main Data Files/response_netcomi"
file_list_otu=list.files(path_otu, pattern="*.csv")
file_list_response=list.files(path_response)
for (i in 1:length(file_list_response)){
        response_file=list.files(file.path(path_response,file_list_response[i]),pattern="^response")
        response=read.csv(file.path(path_response, file_list_response[i],response_file),row.names = 1)
        #response[,1:2]<-NULL
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
                
                # sm<-summary(props_season, numbNodes = 40L)
                # tiff(paste(file.path(path_response, file_list_response[i]),file_list_otu[j],j,".tiff"), width=2300, height=2000, res=300)
                # plot(props_season, 
                #      sameLayout = FALSE, 
                #      nodeColor = "cluster",
                #      #nodeSize = "degree",
                #      nodeSize = "mclr",
                #      rmSingles = TRUE,
                #      cexNodes = 1.5, 
                #      cexLabels = 2.5,
                #      cexHubLabels = 3,
                #      cexTitle = 3.7,
                #      hubBorderCol  = "gray40")
                # dev.off()
                
                
                
                
                
                ###################################Comparing Networkd difference measures
                
                comp_season <- netCompare(props_season, permTest = FALSE, verbose = FALSE)
                
                diff<-summary(comp_season, 
                              groupNames = c("label 0", "label 1"),
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
                k=data.frame(matrix(unlist(m2), nrow=length(m2$label.0.x), byrow=TRUE))
                print(length(m2$label.0.x))
                row.names(k)=row.names(m2)
                colnames(k)<-c("degree0","degree1","dif_degree","bet0","bet1","dif_bit","close0","close1","dif_close")
                write.csv(k,file =file.path(path_response, file_list_response[i],file_list_otu[j]),row.names = TRUE)
                
                
                
                #capture.output(m2, file =file.path(path_response, file_list_response[i],file_list_otu[j]))
                #capture.output(m2, file ='/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Result-total/feature-selection-netcomi/feature-selection-netcomi-no_tuber_scab/feature_diff_label0_1_Phylum.csv')
                
        }
}




#########################################################################################PLOTslide
###################################feature_selection
#path_otu="/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/python code local/Main Data Files/filter_data_na"
unlink("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/python code local/Main Data Files/plot_netcomi/Icon\r")
unlink("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/python code local/Main Data Files/response_netcomi/Icon\r")
path_otu="/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/python code local/Main Data Files/plot_netcomi/"
path_response="/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/python code local/Main Data Files/response_netcomi"
file_list_otu=list.files(path_otu, pattern="*.csv")
file_list_response=list.files(path_response)
for (i in 1:length(file_list_response)){
        response_file=list.files(file.path(path_response,file_list_response[i]),pattern="^response")
        response=read.csv(file.path(path_response, file_list_response[i],response_file),row.names = 1)
        #response[,1:2]<-NULL
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
                
                sm<-summary(props_season, numbNodes = 40L)
                tiff(paste(file.path(path_response, file_list_response[i]),file_list_otu[j],j,".tiff"), width=2300, height=2000, res=300)
                plot(props_season,
                     sameLayout = FALSE,
                     nodeColor = "cluster",
                     nodeSize = "degree",
                     #nodeSize = "mclr",
                     rmSingles = TRUE,
                     cexNodes = 1.5,
                     cexLabels = 2.5,
                     cexHubLabels = 3,
                     cexTitle = 3.7,
                     hubBorderCol  = "gray40")
                dev.off()
                
                
        }}
                
                
             




















###########Last version
#############################################Phylum
data1=read.csv("//Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/CountOTUY1_F_Phylum.csv",header = TRUE,row.names = 1)

##########################################Order
data1=read.csv("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/CountOTUY1_F_Order.csv",header = TRUE,row.names = 1)

############################################Class
data1=read.csv("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/CountOTUY1_F_Class.csv",header = TRUE,row.names = 1)

############################################Family
data1=read.csv("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/CountOTUY1_F_Family.csv",header = TRUE,row.names = 1)

##########################################Genus
data1=read.csv("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/filter_data/CountOTUY1_F_Genus.csv",header = TRUE,row.names = 1)



####################################################
#response=read.csv("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/response.csv",row.names = 1)
#response=read.csv("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/yield_per_meter.csv",row.names = 1)
response=read.csv("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/response/scab_severity/scab_severity.csv",row.names = 1)
response=read.csv("/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Main Data Files/no_tuber_scab.csv",row.names = 1)



response[,1:2]<-NULL
data=merge(data1,response,by="row.names",row.names = 1)
row.names(data)=data[,1]
data[,1]<-NULL
d1=data[data[,dim(data)[2]]==0,]
d2=data[data[,dim(data)[2]]==1,]
colnames(data)[dim(data)[2]]<-"y_binary"
d1['y_binary']<-NULL
d2['y_binary']<-NULL
dim(data)
dim(d1)
dim(d2)

###################################################################
#############################################################
############################################################
###Comparing Networks in zero and one label
net_season <- netConstruct(data = d1, 
                           data2 = d2,  
                           filtTax = "highestVar",
                           filtTaxPar = list(highestVar = min(dim(data)[2],200)),
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

tiff(filename="/Users/rosa/Desktop/Netspring_Family.tiff", width=2300, height=2000, res=300)
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
     hubBorderCol  = "gray40")
dev.off()

###################################Comparing Networkd difference measures

comp_season <- netCompare(props_season, permTest = FALSE, verbose = FALSE)

diff<-summary(comp_season, 
        groupNames = c("label 0", "label 1"),
        showCentr = c("degree", "between", "closeness"), 
        numbNodes = min(100,round(dim(data)[2])))

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
capture.output(m2, file ='/Users/rosa/Desktop/ALLWork/Madison/Project/Soil-nn/Code/pythoncode/Result-total/feature-selection-netcomi/feature-selection-netcomi-no_tuber_scab/feature_diff_label0_1_Phylum.csv')
                


