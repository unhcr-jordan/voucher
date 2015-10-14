
library("boot") ##bootstrapping; cross validation for GLM
library("neuralnet") ##artificial neural network
library("pvclust") ##clustering
library("cluster") ##clustering
library("fpc") ##clustering


######################################################

voucher <- read.csv("out/datamerge.csv")

###unique ration card
uniqueRationCard <- as.data.frame(unique(voucher$Ration.Card))
dim(uniqueRationCard)

###unique barcode
uniqueBarcode <- as.data.frame(unique(voucher$Barcode))

sum(voucher$X..of.vouchers)

#######################################################
###Gender of principal applicant
# $PA.Gender..M.F and $dem.sex are not identical, which is supposed to be?
identical(voucher$PA..Gender.M.F, voucher$dem.sex)

voucher.malePA1 <- voucher[voucher$PA.Gender..M.F == "M",] # male principal applicant  
voucher.femalePA1 <- voucher[voucher$PA.Gender..M.F =="F",] #female principal applicant
voucher.gendernaPA1 <- voucher[voucher$PA.Gender..M.F =="",] #female principal applicant

voucher.malePA2 <- voucher[voucher$dem_sex == "M",] # male principal applicant according to Refugee Registration database
voucher.femalePA2 <- voucher[voucher$dem_sex == "F",] # female principal applicant according to Refugee Registration database

### Gender of Voucher Collector
voucher.maleCollector <-  voucher[voucher$Collector.gender..M.F == "M",] # male collector 
voucher.femaleCollector <-  voucher[voucher$Collector.gender..M.F == "F",] # female collector 

### Family Size
# $Family.Size and $Num_Inds are not identicial, which is supposed to be?
FamilySizeCount1 <- table(voucher$Family.Size)
FamilySizeCount2 <- table(voucher$Num_Inds)
FamilySizeAverage1 <- mean(voucher$Family.Size)
FamilySizeAverage2 <- mean(voucher$Num_Inds)


######################################################
### shopping behavior
sum(voucher$Adult.diapers)
sum(voucher$Disinfectant)
sum(voucher$Food)
sum(voucher$Household.hardware.items)
sum(voucher$Other.hygiene.items)
sum(voucher$Adult.shampoo)
sum(voucher$Baby.diapers)
sum(voucher$Dishwashing.liquid)
sum(voucher$Womens.sanitary.napkins)
sum(voucher$Baby.shampoo)
sum(voucher$Other.items)
sum(voucher$Gas.bottle)
sum(voucher$laundry.soap)
sum(voucher$Laundry.soap)
sum(voucher$Soap.bars)

dim(voucher$Answer.3)

######################################################
###Generalized Linear Model on Answer 3 - "for sale" or "for family use"
purpose.glm <- glm(Answer.3 ~ Family.Size + PA.Gender..M.F. + X..of.vouchers + Num_Inds + admlevel3 
                   + AVG_Age + Child_0_14 + Child_0_17 + Child_0_18 + dem_marriage + dem_age + dem_sex + edu_highest 
                   + Adult.diapers + Disinfectant + Food + Household.hardware.items + Other.hygiene.items + Adult.shampoo + Baby.diapers 
                   + Dishwashing.liquid + Womens.sanitary.napkins + Baby.shampoo + Other.items + Gas.bottle + Laundry.soap + Soap.bars + Median_Age
                   ,data=voucher,family=binomial)
purpose.glm
summary(purpose.glm)

purpose.glm2 <- glm(Answer.3 ~  PA.Gender..M.F. + X..of.vouchers + Num_Inds + edu_highest + dem_marriage + dem_age + dem_sex
                    + Child_0_14 + Child_0_17 + Child_0_18 + Adult.diapers + Food + Household.hardware.items + Other.hygiene.items 
                    + Adult.shampoo + Baby.diapers + Dishwashing.liquid + Other.items + Gas.bottle + Laundry.soap + Soap.bars
                    # + Median_Age + admlevel3 + AVG_Age + Family.Size + Disinfectant + Womens.sanitary.napkins + Baby.shampoo
                    ,data=voucher,family=binomial)
purpose.glm2
summary(purpose.glm2)
anova(purpose.glm,purpose.glm2,test='Chisq')

# five-fold cross-validation
set.seed(17)
purpose.5=rep(0,5)
for(i in 1:5){
  purpose.glm <- glm(Answer.3 ~  PA.Gender..M.F. + X..of.vouchers + Num_Inds + edu_highest + dem_marriage + dem_age + dem_sex
                     + Child_0_14 + Child_0_17 + Child_0_18 + Adult.diapers + Food + Household.hardware.items + Other.hygiene.items 
                     + Adult.shampoo + Baby.diapers + Dishwashing.liquid + Other.items + Gas.bottle + Laundry.soap + Soap.bars
                     # + Median_Age + admlevel3 + AVG_Age + Family.Size + Disinfectant + Womens.sanitary.napkins + Baby.shampoo
                     ,data=voucher,family=binomial)
  purpose.5[i]=cv.glm(voucher,purpose.glm,K=5)$delta[1]
}
purpose.5

######################################################
#########clustering shopping behaviors
### hierarchical agglomerative
purchasingGoods <- c("Adult.diapers","Disinfectant","Food","Household.hardware.items","Other.hygiene.items","Adult.shampoo","Baby.diapers",
                     "Dishwashing.liquid","Womens.sanitary.napkins","Baby.shampoo","Other.items","Gas.bottle","Laundry.soap","Soap.bars")
purchasingData <- voucher[1:5000,purchasingGoods]

d <- dist(purchasingData, method="euclidean")
purchasing.fit <- hclust(d,method="ward.D")
plot(purchasing.fit)
groups <- cutree(purchasing.fit, k=3)
rect.hclust(purchasing.fit,k=3,border="red")

cluster.fit <- pvclust(purchasingData,method.hclust="ward.D",method.dist="euclidean")
plot(cluster.fit)

### k-means 
# preparing data
purchasingData2 <- voucher[purchasingGoods]
purchasingDataNAOmit <- na.omit(purchasingData2)
purchasingDataCleaned <- scale(purchasingDataNAOmit)

#k-means modeling
clustering.fit <- kmeans(purchasingDataCleaned,3)
aggregate(purchasingDataCleaned,by=list(clustering.fit$cluster),FUN=mean)

#visualizing the results
clusplot(purchasingDataCleaned,clustering.fit$cluster,color=TRUE,shade=TRUE,labels=2,lines=0)
plotcluster(purchasingDataCleaned,clustering.fit$cluster)

######################################################
### supervised learning - neural network

### unsupervised learning - anomaly detection