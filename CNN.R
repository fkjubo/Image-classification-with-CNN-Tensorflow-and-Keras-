# importing libraries

library(keras)
library(EBImage)

# setting the work directory

setwd('/Users/Jubo/Documents/Image problem')

# importing the pictures for the train data

pic1 <- c("car1.jpg", "car2.jpg","car4.jpg","car5.jpg","car6.jpg","car8.jpg",
          "cat1.jpg", "cat2.jpg", "cat4.jpg","cat5.jpg","cat6.jpg","cat7.jpg",
          "bike1.jpg","bike2.jpg","bike4.jpg")

# importing the pictures for the test data

pic2 <- c("cat3.jpg", "car3.jpg", "bike3.jpg")

train <- list()

for (i in 1:15) {train[[i]] <- readImage(pic1[i])}

test <- list()

for (i in 1:3) {test[[i]] <- readImage(pic2[i])}

# checking if every variable is working ok

print(train[[6]])
summary(train[[6]])
display(train[[6]])
plot(train[[6]])
str(train)

# resize the images

for(i in 1:15) {train[[i]] <- resize(train[[i]], 100, 100)}
for(i in 1:3) {test[[i]] <- resize(test[[i]], 100, 100)}

# combinig all the images from lists

train <- combine(train)
test <- combine(test)

# displaying all the images by 2 rows for train data

x <- tile(train, 2)
display(x)

# displaying all the images by 1 rows for test data

y <- tile(test, 1)
display(y)

# reorder the dimention of the data

train <- aperm(train, c(4, 1, 2, 3))
test <- aperm(test, c(4, 1, 2, 3))

# creating a response for images

trainResponse <- c(1,1,1,1,1,1,
                   0,0,0,0,0,0,
                   2,2,2)
testResponse <- c(0,1,2)

# one hot encoding

trainLabels <- to_categorical(trainResponse)
testLabels <- to_categorical(testResponse)

# creating the model structure

model <- keras_model_sequential()

model %>%
  layer_conv_2d(filters = 32,
                kernel_size = c(3,3),
                activation = "relu",
                input_shape = c(100,100,3)) %>%
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = .25) %>%
  
  layer_conv_2d(filters = 32,
                kernel_size = c(3,3),
                activation = "relu") %>%
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = .25) %>%
  
  layer_conv_2d(filters = 64,
                kernel_size = c(3,3),
                activation = "relu") %>%
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = .25) %>%
  
  layer_conv_2d(filters = 64,
                kernel_size = c(3,3),
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = .25) %>%
  
  layer_flatten() %>%
  
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(rate = .35) %>%
  layer_dense(units = 3, activation = "softmax")

# compile the model

model %>%
  compile(loss = "categorical_crossentropy",
          optimizer = optimizer_sgd(lr = .002, decay = 1e-6, momentum = .9, nesterov = T),
          metrics = "accuracy")

# fitting data

history <- model %>%
               fit(train,
               trainLabels,
               validation_split = .1,
               epochs = 300,
               #steps_per_epoch = 100,
               #validation_steps = 2,
               batch_size = 32)

plot(history)
  
# prediction

model %>% evaluate(train, trainLabels)

pred <- model %>% predict_classes(train)

prob <- model %>% predict_proba(train)

table(Predicted= pred, actual = trainResponse)
