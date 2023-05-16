# Load required libraries
library(readxl)
library(dplyr)
library(ggplot2)
library(cluster)
library(factoextra)
library(NbClust)
library(scales)

# Read dataset
dataset <- read_excel("vehicles.xlsx")
dataset <- dataset[, -1] # Remove index column

# Pre-processing: Scaling
scaled_dataset <- as.data.frame(scale(dataset[, 1:18]))

# Outliers detection/removal
outliers <- boxplot.stats(scaled_dataset$Comp)$out
scaled_dataset <- scaled_dataset[!(scaled_dataset$Comp %in% outliers),]

# Determine number of cluster centers
nbclust_result <- NbClust(scaled_dataset, distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans")
elbow_result <- fviz_nbclust(scaled_dataset, kmeans, method = "wss")
gap_result <- clusGap(scaled_dataset, FUN = kmeans, nstart = 25, K.max = 10, B = 50)
silhouette_result <- fviz_nbclust(scaled_dataset, kmeans, method = "silhouette")

# K-means clustering
favoured_k <- nbclust_result$Best.nc[1]
kmeans_result <- kmeans(scaled_dataset, centers = favoured_k, nstart = 25)

# Ratio of BSS/TSS
BSS <- sum(kmeans_result$betweenss)
TSS <- sum(kmeans_result$totss)
BSS_TSS_ratio <- BSS / TSS

# WSS
WSS <- kmeans_result$tot.withinss

# Silhouette plot and average silhouette width score
silhouette_plot <- fviz_silhouette(silhouette(kmeans_result$cluster, dist(scaled_dataset)))
avg_sil_width <- mean(silhouette(kmeans_result$cluster, dist(scaled_dataset))[, 3])

# Display results
print(nbclust_result)
print(elbow_result)
print(gap_result)
print(silhouette_result)
print(kmeans_result)
cat("BSS/TSS ratio:", BSS_TSS_ratio, "\n")
cat("WSS:", WSS, "\n")
print(silhouette_plot)
cat("Average silhouette width score:", avg_sil_width, "\n")
