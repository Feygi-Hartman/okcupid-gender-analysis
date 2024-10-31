sink("C:\\R_workes\\big data\\hw3\\Week5_datingNLP.out")
#####################Downloading relevant libraries#############################
library(stringr)
library(dplyr)
library(ggplot2)
library(xtable)
library(gridExtra)
library(stopwords)
library(Matrix)
library(caret)
library(quanteda)
library(tm)
library(SnowballC)
library(rpart)
library(rpart.plot)

# Set rounding to 2 digits
options(digits=2)

#########################Downloading the data###################################
profiles <- read.csv("C:\\R_workes\\big data\\hw3\\okcupid_profiles.csv", header = TRUE, stringsAsFactors = FALSE)
#Defining the sample size
n <- nrow(profiles)

##################Organization and arrangement of the data######################
#select only the essays columns into a new data
essays <- select(profiles, starts_with("essay"))
#merge all essay columns into one string.
essays <- apply(essays, MARGIN = 1, FUN = paste, collapse=" ")

###########################clean the data#######################################
#after looking on the data we put attention that strings length 9 are empty, we will drop them from the data
profiles <- profiles[nchar(essays) != 9, ]
essays <- essays[nchar(essays) != 9]
#update the sample size
n <- nrow(profiles)

#keep the sizes of the strings in essays in order to follow the clean of the data:
essays_sizes <- nchar(essays)
#> head(essays_sizes)
#[1] 2476 1393 5379  449  662 2343

#clean html symbols and some stopwords from the data
html <- c( "<a", "class=.ilink.", "\n", "\\n", "<br ?/>", "/>", "/>\n<br" )
html.pat <- paste0( "(", paste(html, collapse = "|"), ")" )
stop.words <-  c( "a", "am", "an", "and", "as", "at", "are", "be", "but", "can", "do", "for",
                  "have", "i'm", "if", "in", "is", "it", "like", "love", "my", "of", "on", "or", 
                  "so", "that", "the", "to", "with", "you", "i" )
stop.words.pat <- paste0( "\\b(", paste(stop.words, collapse = "|"), ")\\b" )

essays <- str_replace_all(essays, html.pat, " ")
essays <- str_replace_all(essays, stop.words.pat, " ")
essays_sizes1 <- nchar(essays)
#> head(essays_sizes1)
#[1] 2246 1281 5044  428  629 2157
essays <- gsub("[-_.]", " ", essays)
essays_sizes2 <- nchar(essays)




########################### Tokenize essay texts################################
all.tokens <- tokens(essays, what = "word",
                     remove_numbers = TRUE, remove_punct = TRUE,
                     remove_symbols = TRUE)
#keep the number of words in order to follow the clean of the data
words_sum<-sum(sapply(all.tokens, length))
#> words_sum
#[1] 14110116

all.tokens <- tokens_tolower(all.tokens)
all.tokens <- tokens_select(all.tokens, stopwords(),
                            selection = "remove")
words_sum1<-sum(sapply(all.tokens, length))
#> words_sum1
#[1] 11901664

# Perform stemming on the tokens.
all.tokens <- tokens_wordstem(all.tokens, language = "english")
# remove single-word tokens after stemming. Meaningless
all.tokens <- tokens_select(all.tokens, "^[a-z]$",
                            selection = "remove", valuetype = "regex")
words_sum2<-sum(sapply(all.tokens, length))
#> words_sum2
#[1] 11752537

################# Create a document-term frequency matrix#######################
all.tokens.dfm <- dfm(all.tokens)
dimensions<-dim(all.tokens.dfm)
#> dimensions
#[1]  57822 112213

all.tokens.dfm <- dfm_select(all.tokens.dfm, pattern = "^http", selection = "remove", valuetype = "regex")
dimensions1<-dim(all.tokens.dfm)
#> dimensions1
#[1]  57822 112209

features <- featnames(all.tokens.dfm)
#from looking on the data we saw that lond strings are "garbage words".
long_features <- features[nchar(features) > 12]
all.tokens.dfm <- dfm_select(all.tokens.dfm, pattern = long_features, selection = "remove")
dimensions2<-dim(all.tokens.dfm)
#> dimensions2
#[1]  57822 108667

#####################Calculate TF-IDF directly on the DFM ######################
tf <- dfm_weight(all.tokens.dfm, scheme = "count")
# Normalize the term frequencies directly using dfm_weight
tf_normalized <- dfm_weight(tf, scheme = "prop")
df <- docfreq(tf)#מחשב בכמה מסמכים הופיע כל מילה (מחזיר וקטור)
idf <- log(n / (df))
#החישוב של הלוגריתם הוא על בסיס
#e
tfidf <- tf_normalized * idf

#######################remove words with low variance:#########################
# Convert dfm to a sparse matrix
sparse_dfm <- as(tfidf, "dgCMatrix")
n_docs <- nrow(sparse_dfm)
col_means <- colMeans(sparse_dfm)
col_sum_squares <- colSums(sparse_dfm^2)
# Variance calculation using the formula: var = (sum(x^2) - n * mean(x)^2) / (n - 1)
col_variances <- (col_sum_squares - n_docs * col_means^2) / (n_docs - 1)
# Set a threshold for low variance
threshold <- 0.001
high_variance_cols <- names(col_variances[col_variances > threshold])
# Subset the dfm to keep only high variance columns
dfm_high_variance <- sparse_dfm[, high_variance_cols]

###################convert into dataframe and add labels:#######################
profiles$sex <- as.factor(profiles$sex)
# Convert sparse matrix to dense matrix
dfm_high_variance_dense <- as.matrix(dfm_high_variance)
# Convert dense matrix to data frame
high_variance_df <- as.data.frame(dfm_high_variance_dense)

# Add sex labels to the data frame
high_variance_df<-cbind(profiles$sex,high_variance_df)
names(high_variance_df)[names(high_variance_df)=="profiles$sex"]<-"sex_label"
colnames(high_variance_df) <- make.names(colnames(high_variance_df))

#########################remove unnecessary large variables#####################
rm(profiles)
rm(essays)
rm(all.tokens)
rm(all.tokens.dfm)
rm(tf)
rm(df)
rm(idf)
rm(tf_normalized)
rm(dfm_high_variance)
rm(dfm_high_variance_dense)
rm(sparse_dfm)
rm(col_means)
rm(col_sum_squares)
rm(col_variances)
rm(features)
gc()

##################################run a model###################################
# Splitting the dataset into 80% training and 20% testing
set.seed(123)  # For reproducibility
train_index <- createDataPartition(high_variance_df$sex_label, p = 0.9, list = FALSE)
train_data <- high_variance_df[train_index, ]
test_data <- high_variance_df[-train_index, ]

# Train the model on the training set
folds <- createMultiFolds(train_data$sex_label, k = 10, times = 3)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, index = folds)
#train_rpart_model <- train(
#  sex_label ~ ., 
#  data = train_data, 
#  method = "rpart", 
#  trControl = train_control,
#  tuneGrid = data.frame(cp = 0.001),  # ערך cp נמוך
  #control = rpart.control(maxdepth = 0)  # אין הגבלה על עומק העץ
#)
train_time <- system.time({
  train_rpart_model <- train(
    sex_label ~ ., 
    data = train_data, 
    method = "rpart", 
    trControl = train_control,
    tuneGrid = data.frame(cp = 0.001),  # ערך cp נמוך
  )
})
print(train_time)
#Saving the model to facilitate a work sequence
save(train_rpart_model, file = "C:\\R_workes\\big data\\hw3\\train_rpart_model.RData")
#load("C:\\R_workes\\big data\\hw3\\train_rpart_model.RData")
rpart_model<- train_rpart_model$finalModel
#print the tree into a pdf file
pdf(file = "C:\\R_workes\\big data\\hw3\\rpart_model_plot2.pdf")
rpart.plot(rpart_model)#plot the tree 
dev.off()


split_words <- rpart_model$frame[, "var"]
split_words <- split_words[split_words != "<leaf>"]
split_words<-unique(split_words)
sink(file = "C:\\R_workes\\big data\\hw3\\rpart_model_split words.txt")
split_words
sink(file = NULL)

#############test the model by calculating a confusion matrix ##################
# Make predictions on the test set
predictions <- predict(train_rpart_model, test_data)

# Generate the confusion matrix on the test set
confusionMatrix(predictions, test_data$sex_label)
#Confusion Matrix and Statistics

#          Reference
#Prediction    f    m
#         f 1098  723
#         m 1240 2721

##Remove the batch effect: find and eliminate “male words” and “female words”###
#fined lists of "male words" and "fmale words":
tree_frame <- rpart_model$frame

# Get the splits and variable names
split_vars <- rownames(tree_frame[tree_frame$var != "<leaf>", ]) # Nodes with splits
leaf_nodes <- tree_frame[tree_frame$var == "<leaf>", ] # Terminal (leaf) nodes

# Initialize lists for male and female associated words
male_words <- c()
female_words <- c()

# Loop over the split nodes and find out whether they lead to male or female predictions
for (node in split_vars) {
  # Get the word responsible for the split at the current node
  word <- tree_frame[node, "var"]
  
  # Get the left and right child node indices
  left_child <- as.character(as.numeric(node) * 2)
  right_child <- as.character(as.numeric(node) * 2 + 1)
  
  # Check if these child nodes exist in the frame (to avoid the missing value issue)
  if (left_child %in% rownames(tree_frame) & right_child %in% rownames(tree_frame)) {
    
    # Check if the left and right children are leaf nodes
    left_is_leaf <- tree_frame[left_child, "var"] == "<leaf>"
    right_is_leaf <- tree_frame[right_child, "var"] == "<leaf>"
    
    # Check the predicted class in the left and right children (1 for female, 2 for male)
    if (left_is_leaf && tree_frame[left_child, "yval"] == 1) {
      female_words <- c(female_words, word)
    } 
    if (left_is_leaf && tree_frame[left_child, "yval"] == 2) {
      male_words <- c(male_words, word)
    }   
    if (right_is_leaf && tree_frame[right_child, "yval"] == 1) {
      female_words <- c(female_words, word)
    }  
    if (right_is_leaf && tree_frame[right_child, "yval"] == 2) {
      male_words <- c(male_words, word)
    }
  }
}

# Output the male and female associated words
print("Male-associated words:")
print(unique(male_words))

print("Female-associated words:")
print(unique(female_words))

#clean the df from all split words
df_cleaned <- high_variance_df[ , !(colnames(high_variance_df) %in% split_words)]

#############Cluster the applicants to 2,3,4 and 10 clusters (kmeans)###########


# Extract the label column and remove it from the numeric data
df_labels <- df_cleaned$sex_label
df_numeric <- df_cleaned %>% select(-sex_label)  # Exclude the 'sex_label' column

# Scale the numeric data
df_scaled <- scale(df_numeric)

# Function to perform k-means clustering and return the results
perform_kmeans <- function(data, centers) {
  kmeans_result <- kmeans(data, centers = centers, nstart = 25)
  return(kmeans_result)
}
# Number of clusters to try
num_clusters <- c(2,3,4,10)
# List to store k-means results
kmeans_results <- list()
# Perform k-means clustering
for (k in num_clusters) {
  kmeans_results[[paste0("k_", k)]] <- perform_kmeans(df_scaled, k)
}
#Saving the kmeans_results to facilitate a work sequence
saveRDS(kmeans_results, file = "C:\\R_workes\\big data\\hw3\\kmeans_results.rds")
#kmeans_results <- readRDS("kmeans_results.rds")

#################plot T-SNE or PCA of the cluster results#######################

#PCA
pca_result <- prcomp(df_scaled, scale. = TRUE)
pca_data <- data.frame(pca_result$x[,1:2])  # Extract the first two principal components

pdf(file = "C:\\R_workes\\big data\\hw3\\Week5_datingNLP.pdf")
for (k in num_clusters){
  kmeans_cluster <- kmeans_results[[paste0("k_", k)]]$cluster

  pca_data$cluster <- as.factor(kmeans_cluster)  # Add the cluster labels
  
  #Plotting PCA
  pca_plot <- ggplot(pca_data, aes(x = PC1, y = PC2, color = cluster)) +
    geom_point(size = 2) +
    ggtitle(paste("PCA of K-Means Clusters (k =", k, ")")) +
    theme_minimal()
 
   print(pca_plot)
  
}
dev.off()

save(file="C:\\R_workes\\big data\\hw3\\Week5_datingNLP.rdata", train_rpart_model, tfidf, pca_data)


sink(file = NULL)













