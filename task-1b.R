# Load required libraries
library(readxl)     # for reading Excel files
library(factoextra) # for visualization of clustering results
library(NbClust)    # for clustering evaluation using various criteria
library(cluster)    # for clustering algorithms
library(ggplot2)    # for data visualization
library(fpc)        # for fuzzy clustering algorithms

# set a seed for reproducibility
set.seed(42)

# Load the dataset
dataset <- read_excel("vehicles.xlsx")

# Scale the data
scaled_data <- scale(dataset[, 2:19])

# Define a function to remove outliers based on the IQR method
remove_outliers <- function(data, k = 1.5) {
  Q1 <- apply(data, 2, quantile, probs = 0.25)
  Q3 <- apply(data, 2, quantile, probs = 0.75)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - k * IQR
  upper_bound <- Q3 + k * IQR
  return(data[apply(data, 1, function(x) all(x >= lower_bound & x <= upper_bound)), ])
}

# Clean the data by removing outliers
clean_data <- remove_outliers(scaled_data)

# Perform principal component analysis
pca_result <- prcomp(clean_data, scale. = TRUE)
summary(pca_result)

# Calculate cumulative proportion of variance explained by each principal component
cumulative_score <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)

# Determine number of principal components that explain at least 92% of the variance
selected_pcs <- which(cumulative_score >= 0.92)[1]

# Extract the selected principal components from the PCA result
pca_data <- pca_result$x[, 1:selected_pcs]

# NbClust to determine the optimal number of clusters
nbclust_result <- NbClust(pca_data, distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans")
best_k_nbclust <- as.numeric(names(which.max(table(nbclust_result$Best.nc))))

# Set the plotting layout to one row and one column
par(mfrow = c(1, 1))

# Elbow method to determine the optimal number of clusters
elbow_result <- fviz_nbclust(pca_data, kmeans, method = "wss", k.max = 10)
best_k_elbow <- as.numeric(elbow_result$data[which.min(elbow_result$data$y), "clusters"])
plot(elbow_result, frame = FALSE, xlab="Number of clusters K", ylab="Total within-clusters sum of squares")

# Gap statistic to determine the optimal number of clusters
gap_stat <- clusGap(pca_data, FUN = kmeans, nstart = 25, K.max = 10, B = 50)
best_k_gap <- as.numeric(maxSE(gap_stat$Tab[, "gap"], gap_stat$Tab[, "SE.sim"], method = "Tibs2001SEmax"))
fviz_gap_stat(gap_stat)

# Silhouette method to determine the optimal number of clusters
silhouette_result <- fviz_nbclust(pca_data, kmeans, method = "silhouette", k.max = 10)
best_k_silhouette <- as.numeric(silhouette_result$data[which.max(silhouette_result$data$y), "clusters"])
plot(silhouette_result, frame = FALSE,  xlab = "Number of clusters K", ylab = "Average Silhouettes")

# Print the best K values obtained by the different methods
cat("Best K values:\n")
cat("NbClust:", best_k_nbclust, "\n")
cat("Elbow:", best_k_elbow, "\n")
cat("Gap statistic:", best_k_gap, "\n")
cat("Silhouette:", best_k_silhouette, "\n")

# Select the best K value
k_value_table <- table(c(best_k_nbclust, best_k_elbow, best_k_gap, best_k_silhouette))
best_k <- as.numeric(names(k_value_table)[k_value_table == max(k_value_table)])
cat("Selected best K:", best_k, "\n")

# Perform K-means clustering
kmeans_result <- kmeans(pca_data, centers = best_k)
fviz_cluster(kmeans_result, data = pca_data)

# Calculate BSS/TSS ratio
TSS <- sum(kmeans_result$totss)
BSS <- sum(kmeans_result$betweenss)
WSS <- sum(kmeans_result$tot.withinss)
BSS_TSS_ratio <- BSS / TSS

# Print results
cat("BSS:", BSS, "\n")
cat("WSS:", WSS, "\n")
cat("BSS/TSS ratio:", BSS_TSS_ratio, "\n")

# Plot Silhouette plot
sil <- silhouette(kmeans_result$cluster, dist(pca_data))
avg_sil_width <- mean(sil[, 3])
plot(sil, main = paste("Silhouette plot, average width:", round(avg_sil_width, 2)), col = 1:best_k, border=NA)

# Calculate the distance matrix between the principal components
diss <- dist(pca_data)

# Compute cluster statistics using the distance matrix and K-means clustering results
cluster_stats <- cluster.stats(diss, kmeans_result$cluster)

# Obtain the Calinski-Harabasz index from the cluster statistics
calinski_harabasz <- cluster_stats$ch

# Plot the clusters in the principal component space and include the Calinski-Harabasz index in the plot title
clusplot(pca_data, kmeans_result$cluster, color=TRUE, shade=TRUE, labels=2, lines=0, main=paste("Calinski-Harabasz Index =", round(calinski_harabasz,2)))
