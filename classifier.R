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
              "trump",
              "president",
              "scientist",
              "crisis",
              "national",
              "research",
              "science",
              "new",
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
test <- d[split,] %>% dplyr::select(-c(tweet,category))

# --- train Naive Bayes classifier with 40 observations

# calculate priors
pol_prior <- sum(as.numeric(as.character(train$binary)))/nrow(train) #politics
sci_prior <- nrow(train %>% filter(binary==0))/nrow(train) #science

# separate classes
pol_tweets <- train %>% filter(binary == 1)
sci_tweets <- train %>% filter(binary == 0)

# P( keyword | politics/science )
probability_table <- list()
for(i in c(keywords)){
  pt <- data.frame(keyword= i,
                   politics= sum(pol_tweets[,i])/nrow(pol_tweets), #P(keyword|politics)
                   science= sum(sci_tweets[,i])/nrow(sci_tweets)) #P(keyword|science)
  probability_table <- append(probability_table,list(pt))
}

probability_table <- do.call(rbind, probability_table) %>% arrange(keyword)
# divide up the above table into P(x|politics) and P(x|science)
pol_table <- probability_table %>% select(keyword,politics)
sci_table <- probability_table %>% select(keyword,science)
# THESE ARE OUR TRAINING PROBABILITIES! 

# --------------- P(politics | Keyword) --------------- #
# calculate by -- P(keyword1 | +) * P(keyword2 | +) * .... * P(politics)

# initialize a new dataset for P(x|politics) results
pol_test <- test
# replace binary features with P(x|politics) from training (pol_table)
for(feat in keywords){
  pol_test[,feat] <- pol_test[,feat] * pol_table[pol_table$keyword==feat,2]
}

mult <- pol_test
mult[mult == 0] <- 1 #change all 0s to 1 b/c we are multiplying row-wise to compute probabilities
mult$prior <- rep(pol_prior,nrow(pol_test))
mult <- mult %>% dplyr::select(-binary)
# multiply non zero entries across keywords to get P(political|keyword)!
pol_post <- rowProds(as.matrix(mult))
pol_test$pol_post <- pol_post

# --------------- P(science | Keyword) --------------- #
# calculate by -- P(keyword1 | -) * P(keyword2 | -) * .... * P(science)

# initialize a new dataset for P(x|science) results
science_test <- test
# replace binary features with P(x|science) from training
for(feat in c(keywords)){
  science_test[,feat] <- science_test[,feat] * sci_table[sci_table$keyword==feat,2]
}

# multiply non zero entries across keywords to get P(science|keyword)!
mult <- science_test
mult[mult == 0] <- 1 #change all 0s to 1 b/c we are multiplying row-wise to compute probabilities
mult$prior <- rep(sci_prior,nrow(science_test))
mult <- mult %>% dplyr::select(-binary)
# multiply non zero entries across keywords to get P(political|keyword)!
sci_post <- rowProds(as.matrix(mult))
science_test$sci_post <- sci_post

# --------------- Predict Tweet Category --------------- #
# if pos_prob > neg_prob, predict as 1 (else, label as 0)

final_table <- data.frame(binary = test$binary,
                          politics_post = pol_post,
                          science_post = sci_post,
                          predicted = rep(1,nrow(test)))

for(i in 1:nrow(final_table)){
  if(final_table[i,"politics_post"] < final_table[i,"science_post"]){
    final_table[i,"predicted"] <- 0
  }
}

# --------------- model performance --------------- #

performance_suite <- function(labels,prediction){
  
  test_table <- table(labels,prediction)
  print(test_table)
  print("% Accuracy")
  acc <- (test_table[1,1]+test_table[2,2])/sum(test_table)*100
  print(acc)
  # precision
  p <- (test_table[2,2])/(test_table[2,2]+test_table[1,2]) 
  print("Precision")
  print(p)
  # recall
  r <- (test_table[2,2])/(test_table[2,2]+test_table[2,1]) 
  print("Recall")
  print(r)
  # f1
  f1 <- (2*r*p)/(r+p)
  print("F1 Score")
  print(f1)
  # ROC curve
  pred <- prediction(predictions=as.numeric(prediction),labels=labels)
  roc.perf <- performance(pred, measure = "tpr", x.measure = "fpr")
  pdf("output/ROCcurve.pdf")
  print(plot(roc.perf,main="ROC Curve") )
  print(abline(a=0, b= 1,col="red",lty=2))
  dev.off()
}

labels <- final_table$binary
predictions <- final_table$predicted

performance_suite(labels,predictions)
