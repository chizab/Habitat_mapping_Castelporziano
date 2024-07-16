#Random Forest script

setwd("")

#load required libraries
library(raster)
library(sf)
library(sp)
library(caret)
library(randomForest)
library(data.table)
library(dplyr)
library(stringr)
library(doParallel)
library(snow)
library(parallel)


rasterOptions(tmpdir = "./cache/temp") # set the temporary folder for raster package operations

raster  <- stack("raster.tif") #upload of raster image

names(raster) <- c("...","...") #set names of the layers in the stack

plot(raster) #plot for visual check

points_training <- st_read("training.shp") #import training points
points_training$id <- as.integer(factor(points_training$EUNIS_class)) #give a numeric ID to each class

dt_training <- raster %>% 
  extract(y = points_training) %>% 
  as.data.table %>% 
  mutate(id = points_training$id) #extract values from raster stack for each training point

#give class name to each class ID
dt_training$id[dt_training$id == 1] <- "C2"
dt_training$id[dt_training$id == 2] <- "C3"
dt_training$id[dt_training$id == 3] <- "C4"
dt_training$id[dt_training$id == 4] <- "N1G"
dt_training$id[dt_training$id == 5] <- "C5"
dt_training$id[dt_training$id == 6] <- "C6"
dt_training$id[dt_training$id == 7] <- "C7"
dt_training$id[dt_training$id == 8] <- "S51"
dt_training$id[dt_training$id == 9] <- "C8"
dt_training$id[dt_training$id == 10] <- "T195"
dt_training$id[dt_training$id == 11] <- "T19B6"
dt_training$id[dt_training$id == 12] <- "T211"
dt_training$id[dt_training$id == 13] <- "T212"

dt_training$EUNIS_2021 <- as.factor(dt_training$id)

points_test <- st_read("test.shp") #same procedure for validation dataset
points_test$id <- as.integer(factor(points_test$Simpl_2))

dt_test <- raster %>% 
  extract(y = points_test) %>% 
  as.data.table %>% 
  mutate(id = points_test$id)

dt_test$id[dt_test$id == 1] <- "C2"
dt_test$id[dt_test$id == 2] <- "C3"
dt_test$id[dt_test$id == 3] <- "C4"
dt_test$id[dt_test$id == 4] <- "N1G"
dt_test$id[dt_test$id == 5] <- "C5"
dt_test$id[dt_test$id == 6] <- "C6"
dt_test$id[dt_test$id == 7] <- "C7"
dt_test$id[dt_test$id == 8] <- "S51"
dt_test$id[dt_test$id == 9] <- "C8"
dt_test$id[dt_test$id == 10] <- "T195"
dt_test$id[dt_test$id == 11] <- "T19B6"
dt_test$id[dt_test$id == 12] <- "T211"
dt_test$id[dt_test$id == 13] <- "T212"

dt_test$EUNIS_2021 <- as.factor(dt_test$id)

dt_training <- dt_training[,-c(...)] #remove unnecessary columns
dt_test <- dt_test[,-c(...)]

tr_fix <- na.omit(dt_training) #omit eventual NA values
te_fix <- na.omit(dt_test)

# create cross-validation folds
n_folds <- 10
set.seed(321)
folds <- createFolds(1:nrow(tr_fix), k = n_folds)
# Set the seed at each resampling iteration
seeds <- vector(mode = "list", length = n_folds + 1) # +1 for the final model
for(i in 1:n_folds) seeds[[i]] <- sample.int(1000, n_folds)
seeds[n_folds + 1] <- sample.int(1000, 1) # seed for the final model

control <- trainControl(summaryFunction = multiClassSummary,
                        method = "cv",
                        number = n_folds,
                        search = "grid",
                        classProbs = TRUE, 
                        savePredictions = TRUE,
                        index = folds,
                        seeds = seeds)

#RANDOM FOREST

cluster <- makeCluster(3/4 * detectCores())
registerDoParallel(cluster)
model_rf <- caret::train(EUNIS_2021 ~ . , method = "rf", data = tr_fix,
                         importance = TRUE, 
                         allowParallel = TRUE,
                         tuneGrid = data.frame(mtry = c(...,...)),
                         trControl = control)
stopCluster(cluster); remove(cluster)

registerDoSEQ()
saveRDS(model_rf, file = "./....rds") #save RDS file

model_rf$times$everything # total computation time

plot(model_rf) #plot RF accuracies according to mtry parameter

cm_rf <- confusionMatrix(data = predict(model_rf, newdata = te_fix),
                         te_fix$EUNIS_2021) #compute confusion matrix
cm_rf #plot confusion matrix in the console

model_rf$finalModel #plot 10-fold cross-validation results

rf_class <-raster::predict(model = model_rf, object= raster, type= "raw" ) #classification map

writeRaster(rf_class, "rf_class", format='GTiff') #save classification map at geotiff in work directory

rf_prob <-raster::predict(model = model_rf, object= raster, type= "prob", index = 1:13) #probability map for each class

plot(rf_prob, colour=palette, 1:13) #plot probability map to check that each layer is present

writeRaster(rf_prob, "rf_prob", format='GTiff', overwrite=TRUE) #save probability map; result will be a geotiff with a layer for each class

varImp <- varImp(model_rf, scale=TRUE) #generates per-class variable importance

varImp #plots per-class variable importance in the console (numeric)

plot(varImp) #plots per-class variable importance (graphic)
