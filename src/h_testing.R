##################################################################
# A Meta-Analysis of Research in Random Forests for Classification
##################################################################

# required packages from bioconductor for scmamp package
source("https://bioconductor.org/biocLite.R")
biocLite("graph")
a
biocLite("Rgraphviz")
a

# intall stats package
install.packages('scmamp')

# load package
library(scmamp)

#####################################
# Test differnces in generalisation #
#####################################

# set directory to testing files
setwd('/media/canyon/170b9e40-9ed0-4cc6-a29c-797ed4cf77ed/repos/noisy_relu_shift/h_test_data')

# read in data for generalisation
mnist_0 <- read.csv("mnist_none_0_white_list_generalisation.csv")
mnist_5 <- read.csv("mnist_dropout_0.5_white_list_generalisation.csv")
mnist_7 <- read.csv("mnist_dropout_0.7_white_list_generalisation.csv")
mnist_9 <- read.csv("mnist_dropout_0.9_white_list_generalisation.csv")
cifar10_0 <- read.csv("cifar10_none_0_white_list_generalisation.csv")
cifar10_5 <- read.csv("cifar10_dropout_0.5_white_list_generalisation.csv")
cifar10_7 <- read.csv("cifar10_dropout_0.7_white_list_generalisation.csv")
cifar10_9 <- read.csv("cifar10_dropout_0.9_white_list_generalisation.csv")
all_gen_test_mat <- rbind(mnist_0, mnist_5, mnist_7, mnist_9, cifar10_0, cifar10_5, cifar10_7, cifar10_9)

# form final testing matrix
test_mat <- all_gen_test_mat[,2:10]

# remove incomplete cases
test_mat_com <- test_mat[complete.cases(test_mat),]

# find excluding rows
remove_indices <- NULL
count <- 1
for(i in 1:nrow(test_mat_com)){
  if(sum(test_mat_com[i,] < 0) > 1){
    remove_indices[count] <- i
    count = count + 1
  }
}
if(!is.null(remove_indices)){
  final_test_mat = test_mat_com[-remove_indices,]
} else {
  final_test_mat = test_mat_com
}

# omnibus tests
imanDavenportTest(final_test_mat)
friedmanAlignedRanksTest(final_test_mat)
quadeTest(final_test_mat)

# post-hoc tests
postHocTest(final_test_mat, test = "friedman", control = "init_4", correct = "finner", alpha = 0.05)

#####################################
# Test differnces in training speed #
#####################################

# set directory to testing files
setwd('/media/canyon/170b9e40-9ed0-4cc6-a29c-797ed4cf77ed/repos/noisy_relu_shift/h_test_data/training_speed3')

# read in data for generalisation
train_mnist_0 <- read.csv("0.3_mnist_none_0_white_list_training_speed.csv")
train_mnist_5 <- read.csv("0.3_mnist_dropout_0.5_white_list_training_speed.csv")
train_mnist_7 <- read.csv("0.3_mnist_dropout_0.7_white_list_training_speed.csv")
train_mnist_9 <- read.csv("0.3_mnist_dropout_0.9_white_list_training_speed.csv")
train_cifar10_0 <- read.csv("0.3_cifar10_none_0_white_list_training_speed.csv")
train_cifar10_5 <- read.csv("0.3_cifar10_dropout_0.5_white_list_training_speed.csv")
train_cifar10_7 <- read.csv("0.3_cifar10_dropout_0.7_white_list_training_speed.csv")
train_cifar10_9 <- read.csv("0.3_cifar10_dropout_0.9_white_list_training_speed.csv")
all_train_test_mat <- rbind(train_mnist_0, train_mnist_5, train_mnist_7, train_mnist_9, 
                          train_cifar10_0, train_cifar10_5, train_cifar10_7, train_cifar10_9)

# form final testing matrix
train_test_mat <- all_train_test_mat[,2:10]

# remove incomplete cases
train_test_mat_com <- train_test_mat[complete.cases(train_test_mat),]

# find excluding rows
remove_indices <- NULL
count <- 1
for(i in 1:nrow(train_test_mat_com)){
  if(sum(train_test_mat_com[i,] < 0) > 1){
    remove_indices[count] <- i
    count = count + 1
  }
}
if(!is.null(remove_indices)){
  final_train_test_mat = train_test_mat_com[-remove_indices,]
} else {
  final_train_test_mat = train_test_mat_com
}

# omnibus tests
imanDavenportTest(final_train_test_mat)
friedmanAlignedRanksTest(final_train_test_mat)
quadeTest(final_train_test_mat)

# post-hoc tests
postHocTest(final_train_test_mat, test = "friedman", control = "init_4", correct = "finner", alpha = 0.05)
