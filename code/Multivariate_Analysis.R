# Multivariate Socio-Economic Analysis of Deprivation in Ireland
# Valeria Garcia Hernandez 

set.seed(1234)
pobal_index_set = read.csv("pobal_di_2022.csv")
dim(pobal_index_set)

# plot marginal distributions
par(mfrow=c(3,5), mar=c(4, 4, 4, 1)) #(3x5 grid)
for(i in 3:15) {
  hist(pobal_index_set[,i], main=names(pobal_index_set[i]), xlab=names(pobal_index_set[i]))
}

# standardize the data by subtracting the mean and dividing by the std. dev. of each variable
numeric_data = pobal_index_set[,3:15]
mean = apply(numeric_data, 2, mean) #compute column means
std_dev = apply(numeric_data, 2, sd) #compute column std. deviations
# standardize (z-score)
numeric_scaled = sweep(numeric_data, 2, mean, "-")
numeric_scaled = sweep(numeric_scaled, 2, std_dev, "/")

# scatter plot matrices
pairs(numeric_scaled[,1:4])
pairs(numeric_scaled[,5:8])
pairs(numeric_scaled[,9:13])

# perform hierarchical clustering
library(dendextend)
# there are 6 deprivation index labels
num_labels = 6
hcl = hclust(dist(numeric_scaled), method="average")

# convert the hcl object into a dendrogram object and colour branches and labels according to clusters
hcl_dend = as.dendrogram(hcl)
par(mfrow = c(1,1))
hcl_dend = set(hcl_dend, "labels_colors", k=num_labels, value=seq(num_labels))
hcl_dend = set(hcl_dend, "branches_k_color", k=num_labels, value=seq(num_labels))
plot(hcl_dend, main="Socio-Economic Data Hierarchical Clustering")

# cut the tree
hcl = cutree(hcl, k=num_labels)

# compare with the deprivation index label
table1 = table(hcl, pobal_index_set$index_lab)
table1
library(mclust)
ARI_hcl = adjustedRandIndex(hcl, pobal_index_set$index_lab) 
ARI_hcl #ARI = 0.0026

# perform k-means
# run the k-means algorithm over the range of values k=1 to 10
WGSS = rep(0, 10)
for(k in 1:10) {
  WGSS[k] = sum(kmeans(numeric_scaled, centers=k)$withinss)
}
plot(1:10, WGSS, type="b", xlab="k", ylab="within group sum of squares", main="Elbow Plot for Choosing k")

# run k-means 10 times with different random starting points and keep the best solution
kmeans_cl = kmeans(numeric_scaled, centers=num_labels, nstart=10)

# compare with the deprivation index label
table2 = table(kmeans_cl$cluster, pobal_index_set$index_lab)
table2
ARI_k = adjustedRandIndex(kmeans_cl$cluster, pobal_index_set$index_lab)
ARI_k #ARI = 0.31

# split the data into a calibration (70%) and validation (30%) sample
calibration_index = sample(1:nrow(numeric_data), size=0.7*nrow(numeric_data), replace=FALSE)
calibration_data = numeric_data[calibration_index, ]
validation_data = numeric_data[-calibration_index, ]

# compute scaling parameters from only the calibration data
train_means = apply(calibration_data, 2, mean)
train_sds = apply(calibration_data, 2, sd)

# scale calibration data
calibration_data = scale(calibration_data, center=train_means, scale=train_sds)

# scale validation data
validation_data = scale(validation_data, center=train_means, scale=train_sds)

# run k-means on calibration data
km_calibration = kmeans(calibration_data, centers=num_labels, nstart=10)

# get centroids
centers = km_calibration$centers

# function to assign nearest centroid
assign_cluster <- function(data, centers) {
  dists_to_c = apply(centers, 1, function(c) sum((data-c)^2))
  which.min(dists_to_c)
}

# assign each validation data to nearest centroid
validation_clusters = apply(validation_data, 1, assign_cluster, centers=centers)

# cluster the validation sample
km_validation = kmeans(validation_data, centers=num_labels, nstart=10)

# compare the two clustering solutions
table(validation_clusters, km_validation$cluster)
ARI_valid = adjustedRandIndex(validation_clusters, km_validation$cluster)
print(ARI_valid) #ARI=0.39

# create a new column for the simplified version of index_lab
pobal_index_set$index_lab_simp = ifelse(pobal_index_set$index_rel >= 0,1,-1)

# LDA
library(MASS)
data_lda = data.frame(
  index_lab_simp = pobal_index_set$index_lab_simp,
  numeric_scaled
)

# perform classification using LDA with leave-one-out cross-validation 
lda = lda(index_lab_simp ~ TOTPOP+AGEDEP+EDLOW+EDHIGH+HLPROF+LCLASS+PEROOM+LONEPA+UNEMPM+UNEMPF+OWNOCC+PRRENT+LARENT, CV=TRUE, data=data_lda)

# compare predictions to true labels and calculate accuracy
tab = table(lda$class, data_lda$index_lab_simp)
accuracy = sum(diag(tab))/sum(tab)
print(accuracy) #model correctly predicts 93% of observations

#classes are roughly balanced
table(pobal_index_set$index_lab_simp)

# perform classification using QDA with leave-one-out cross-validation
qda = qda(index_lab_simp ~ TOTPOP+AGEDEP+EDLOW+EDHIGH+HLPROF+LCLASS+PEROOM+LONEPA+UNEMPM+UNEMPF+OWNOCC+PRRENT+LARENT, CV=TRUE, data=data_lda)

# calculate accuracy
accuracy_qda = sum(diag(table(qda$class, data_lda$index_lab_simp)))/sum(table(qda$class, data_lda$index_lab_simp))
print(accuracy_qda) #QDA accuracy = 0.886

# perform PCA on the standardized data
# since the data has already been standardized, set center and scale = FALSE
PCA = prcomp(numeric_scaled, center = FALSE, scale. = FALSE) 
summary(PCA)

# proportion of variance explained 
proportion_var_explained = (PCA$sdev^2) / sum(PCA$sdev^2)

# Scree Plot
plot(proportion_var_explained, type="b", main="Scree Plot", xlab="PC", ylab="proportion of variance")

# plot PC scores for observations
library(scatterplot3d)
scatterplot3d(PCA$x[,1:3],main="PCA Scores", xlab="PC1", ylab="PC2", zlab="PC3")

# we plot the first two columns as they capture about 70% of the variance in the data
plot(PCA$x[,1], PCA$x[,2], main="PCA Scores (PC1 and PC2)", xlab="PC1", ylab="PC2")

# perform k-means for PCA scores
kmeans_pca = kmeans(PCA$x[,1:3], centers=num_labels, nstart=10)
ARI_pca = adjustedRandIndex(kmeans_pca$cluster, pobal_index_set$index_lab)
ARI_pca # ARI = 0.346

# refit LDA using the first two PC
data_lda_pca = data.frame(
  PCA$x[ ,1:2],
  index_lab_simp = pobal_index_set$index_lab_simp
)

# perform classification using LDA 
lda_pca_model = lda(index_lab_simp ~ PC1+PC2, data=data_lda_pca)

# function for plotting the boundary
boundary <- function(model, data, class = NULL, predict_type = "class",
                     resolution = 100, showgrid = TRUE, ...) {
  
  if(!is.null(class)) cl <- data[,class] else cl <- 1
  data <- data[,1:2]
  k <- length(unique(cl))
  plot(data,  col = ifelse(cl == 1, "green", "red"), pch = ifelse(cl == 1, 16, 17), ...)
  
  r <- sapply(data, range, na.rm = TRUE)
  xs <- seq(r[1,1], r[2,1], length.out = resolution)
  ys <- seq(r[1,2], r[2,2], length.out = resolution)
  g <- cbind(rep(xs, each=resolution), rep(ys, time = resolution))
  colnames(g) <- colnames(r)
  g <- as.data.frame(g)
  
  p <- predict(model, g, type = predict_type)
  if(is.list(p)) p <- p$class
  p <- as.factor(p)
  
  if(showgrid) points(g, col = as.integer(p)+1L, pch = ".")
  
  z <- matrix(as.integer(p), nrow = resolution, byrow = TRUE)
  contour(xs, ys, z, add = TRUE, drawlabels = FALSE,
          lwd = 2, levels = (1:(k-1))+.5)
  
  invisible(z)
}

par(pin = c(3, 2))

# plot showing the linear Bayes decision boundary
boundary(lda_pca_model, data_lda_pca, class="index_lab_simp", main="LDA")

# fit PCR
# the goal is to predict index_rel
library(pls)
data_pcr = data.frame(
  index_rel = pobal_index_set$index_rel,
  numeric_data
)
PCR = pcr(index_rel ~ TOTPOP+AGEDEP+EDLOW+EDHIGH+HLPROF+LCLASS+PEROOM+LONEPA+UNEMPM+UNEMPF+OWNOCC+PRRENT+LARENT, 
          data=data_pcr, scale=TRUE, validation="CV"
)

# plot MSEP for the different number of components
validationplot(PCR, val.type="MSEP")

# get the performance at 2 components
msep_values = MSEP(PCR)$val
msep_values[1,1,3]

rmse = sqrt(msep_values[1,1,3])
rmse