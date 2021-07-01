# Data-Migration
This repository contains codes for the migration of Soil Health Card and Farmpulse data from S3 to an SQL database.
- fp_size_migration: generates size of Farmpulse dataframe
- shc_size_migration: generates size of Soil Health Card dataframe
- transfer_multi: transfers all files from one bucket to another using multiprocessing
  - Run this code on SageMaker if session required is more than an hour.
- data_improv_edited: edited Michael's dataimprov notebook to download Farmpulse data from SQL and replace deprecated code.
