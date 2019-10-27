# About: trains a simple Naive Bayes classifier by populating a dataframe with probabilities 

#  ---- initializing environment - uncomment 4 lines below and run if packages are not installed

#install.packages("dplyr")
#install.packages("stringr")
#install.packages("ggplot2")
#install.packages("ROCR")
library(dplyr)
library(stringr)
library(ggplot2)
library(ROCR)
d <- read.csv("tweets.csv")

# ---- set up science or politics classes (name it "binary", class: factor)

d$binary <- rep(1,nrow(d)) # set up new column
d$binary[d$category == "science"] <- 0 # label "science" tweets to 0; politics stay as 1
d$binary <- as.factor(d$binary) # change class to factor
names(d)[1] <- "tweet"

# ---- create categorical variables for keywords (1 if present, 0 otherwise)

# choose keywords here
keywords <- c("climate",
              "change",
              "president",
              "crisis",
              "research",
              "science",
              "environment",
              "health")
# lowercase the tweets -- makes string search below accurate
d$tweet <- tolower(d$tweet)

# set up binary columns for each keyword - label as 1 if keyword is present in a tweet
for(word in keywords){
  d[[word]] <- as.numeric(str_detect(d$tweet,word))
}


# ---- train and test data split

# 80/20 split
set.seed(1)
split <- sample(nrow(d)*0.8,(nrow(d)-nrow(d)*0.8))
train <- d[-split,]
test <- d[split,]

# --- train Naive Bayes classifier with 40 observations

# calculate priors
prior_1 <- sum(as.numeric(as.character(train$binary)))/nrow(train) #politics
prior_0 <- nrow(train %>% filter(binary==0))/nrow(train) #science

# separate classes
class_1 <- train %>% filter(binary == 1)
class_0 <- train %>% filter(binary == 0)

# P( keyword | positive/negative )
probability_table <- list()
for(i in c(keywords)){
  pt <- data.frame(keyword= i,
                   politics= sum(class_1[,i])/nrow(class_1), #P(keyword|politics)
                   science= sum(class_0[,i])/nrow(class_0)) #P(keyword|science)
  probability_table <- append(probability_table,list(pt))
}

probability_table <- do.call(rbind, probability_table) %>% arrange(keyword)
# divide up the above table into P(x|politics) and P(x|science)
pol_table <- probability_table %>% select(keyword,politics)
sci_table <- probability_table %>% select(keyword,science)

# --- classify the test data reviews

rownames(test) <- 1:nrow(test) # row indices were random numbers...change to a regular sequence 1-10



# --------------- P(politics | Keyword) --------------- #
# calculate by -- P(keyword1 | +) * P(keyword2 | +) * .... * P(politics)

# initialize a new dataset for P(x|politics) results
politics_test <- test
# replace binary features with P(x|politics) from training
for(feat in c(keywords)){
  politics_test[,feat] <- politics_test[,feat] * pol_table[pol_table$keyword==feat,2]
}
