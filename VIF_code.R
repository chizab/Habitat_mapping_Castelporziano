#script for VIF computation

setwd("")

library(usdm)
library(raster)

Training_values <- read.csv("Training_values.csv") #input file is a .csv with the training values 
                                                   #of the raster stack with the top 10 indices by MDA score
vifstep(x=Training_values, th = 5, keep = NULL, method = 'pearson')

#now remove layers with VIF value > 5

raster <- stack("raster.tif") #raster stack with the top 10 indices by MDA score

raster_VIF <- dropLayer(raster, c(n,n,...)) #remove the layers with VIF > 5 by progressive numeration

plot(raster_VIF) #plot for visual check

writeRaster(raster_VIF, filename = "raster_VIF", format = "GTiff") #save the raster stack with only VIs with VIF < 5
