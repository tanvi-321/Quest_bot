# Quest Bot

Quest Bot is a Flask-based web application designed to interact with users and provide answers based on uploaded files. The bot can handle file uploads (PDFs, images), process them, and provide detailed responses. It integrates with AWS S3, DynamoDB, and the AWS Textract service.Generate the quiz based on the bot responses of user question to test the knowledge of the user.

## Features

- **Bot Responses**: Fetch responses based on previous interactions from DynamoDB.
- **File Upload**: Upload files (PDF, PNG, JPEG) and store them in AWS S3.
- **Text Extraction from PDF**: Convert PDF documents to text for further analysis.
- **Quiz Mode**: Automatically generate questions from bot responses and track quiz scores.
- **AWS Integration**: Uses AWS DynamoDB, S3, and Textract for data processing.

## Prerequisites

Make sure you have the following installed:

- [Python 3.10+](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/)
- [AWS Account](https://aws.amazon.com/)
  - AWS S3 and DynamoDB access setup.
 
## Create and Activate Virtual Environment
- python -m venv venv
# On Windows
- .\venv\Scripts\activate
# On macOS/Linux
- source venv/bin/activate

## Install Required Packages
- pip install -r requirements.txt

## Run the Flask Application
- flask run --debug


