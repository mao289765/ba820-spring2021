# About

### Data URL:

http://34.86.144.106:8501/

### Data

The data are in a new dataset via GCP called `

At [this URL](https://console.cloud.google.com/bigquery?d=SMSspam&p=questrom&page=dataset) You should be able to see a nested Dataset called SMSspam.  This is the database for the challenge

The dataset (i.e. Database) has 3 tables:

- train / test / sample-submission

> `SELECT * FROM `questrom.SMSspam.sample-submission` LIMIT 10`

Train is the dataset to fit your model, test has the features but lacks a target, and sample submission is a reference file so that can debug your submissions against a valid file.

