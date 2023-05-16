# Load required libraries
library(readxl) # for reading Excel files
library(factoextra) # for visualization of clustering results
library(NbClust) # for clustering evaluation using various criteria
library(cluster) # for clustering algorithms
library(ggplot2) # for data visualization

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

# NbClust to determine the optimal number of clusters
nbclust_result <- NbClust(clean_data, distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans")
best_k_nbclust <- as.numeric(names(which.max(table(nbclust_result$Best.nc))))

# Set the plotting layout to one row and one column
par(mfrow = c(1, 1))

# Elbow method to determine the optimal number of clusters
elbow_result <- fviz_nbclust(clean_data, kmeans, method = "wss", k.max = 10)
best_k_elbow <- as.numeric(elbow_result$data[which.min(elbow_result$data$y), "clusters"])
plot(elbow_result, frame = FALSE, xlab="Number of clusters K", ylab="Total within-clusters sum of squares")

# Gap statistic to determine the optimal number of clusters
gap_stat <- clusGap(clean_data, FUN = kmeans, nstart = 25, K.max = 10, B = 50)
best_k_gap <- as.numeric(maxSE(gap_stat$Tab[, "gap"], gap_stat$Tab[, "SE.sim"], method = "Tibs2001SEmax"))
fviz_gap_stat(gap_stat)

# Silhouette method to determine the optimal number of clusters
silhouette_result <- fviz_nbclust(clean_data, kmeans, method = "silhouette", k.max = 10)
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
kmeans_result <- kmeans(clean_data, centers = best_k)
fviz_cluster(kmeans_result, data = clean_data)

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
sil <- silhouette(kmeans_result$cluster, dist(clean_data))
avg_sil_width <- mean(sil[, 3])
plot(sil, main = paste("Silhouette plot, average width:", round(avg_sil_width, 2)), col = 1:best_k, border=NA)
