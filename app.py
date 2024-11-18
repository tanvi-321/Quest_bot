from dotenv import load_dotenv
from flask import Flask, request, jsonify, Blueprint, Response, current_app
from botocore.exceptions import ClientError
from botocore.exceptions import NoCredentialsError
from botocore.exceptions import ClientError
from datetime import datetime, timezone
from werkzeug.utils import secure_filename
import os
import json
import base64
import boto3
import re
import time
import io
import openai
import uuid
import random

load_dotenv()

aws_region_hack = os.environ.get("hack_aws_region")
aws_access_key_hack = os.environ.get("hack_aws_access_key")
aws_secret_key_hack = os.environ.get("hack_aws_secret_key")
openai.api_key = os.environ.get("OPENAI_API_KEY")

quest_bucket = os.getenv("quest_bucket")


textract_bucket = os.getenv("textract_bucket")

s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_hack,
    aws_secret_access_key=aws_secret_key_hack,
    region_name=aws_region_hack,
)

textract_client = boto3.client(
    "textract",
    aws_access_key_id=aws_access_key_hack,
    aws_secret_access_key=aws_secret_key_hack,
    region_name=aws_region_hack,
)
brt = boto3.client(
    # "kendra",
    service_name="bedrock-runtime",
    aws_access_key_id=aws_access_key_hack,
    aws_secret_access_key=aws_secret_key_hack,
    region_name=aws_region_hack,
)

app = Flask(__name__)

# database integration
# --------------------

# Initialize DynamoDB resource
dynamodb = boto3.resource(
    "dynamodb",
    aws_access_key_id=aws_access_key_hack,
    aws_secret_access_key=aws_secret_key_hack,
    region_name=aws_region_hack,
)

# Connect to the table
table_name = "ChatHistory"
table = dynamodb.Table(table_name)


ALLOWED_EXTENSIONS = {
    "pdf",
    "png",
    "jpeg",
    "jpg",
}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# In-memory storage
questions_store = {}
score_store = {}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_type(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower()


# read the file from quest bucket
def read_pdfs(pdf_path):
    response = s3.get_object(Bucket=quest_bucket, Key=pdf_path)
    file_content = response["Body"].read()
    return file_content


def get_pdfs(pdf_path):
    response = s3.get_object(Bucket=textract_bucket, Key=pdf_path)
    file_content = response["Body"].read()
    return file_content


def count_tokens(text):
    # Use your preferred tokenization strategy here
    return len(text.split())


def upload_to_s3(
    file_name,
    quest_bucket=quest_bucket,
    object_name=None,
):
    try:
        # Generate object name if not provided
        if object_name is None:
            original_filename = file_name.filename
            object_name = re.sub(r"\s", "", original_filename)  # Remove spaces
            print(f"Modified object name: {object_name}")

        # Check if the file type is allowed
        # Check if the file type is allowed
        if not allowed_file(file_name.filename):
            allowed_types = ", ".join(ALLOWED_EXTENSIONS)
            return {
                "error": f"Invalid file type. Supported types are: {allowed_types}"
            }, 400

        # Check if the file size exceeds the limit
        if file_name.content_length > MAX_FILE_SIZE:
            return {"error": "File size exceeds the maximum limit (5MB)"}, 400

        # Check the content type before attempting upload
        if file_name.content_type not in ["application/pdf", "image/png", "image/jpeg"]:
            return {"error": "Unsupported file type"}, 400

        # Upload the file to S3
        s3.upload_fileobj(file_name, quest_bucket, object_name)

        print("File uploaded successfully to S3")
        print("AWS Region:", aws_region_hack)

        # Get the file type for further processing
        file_type = get_file_type(object_name)
        print(file_type)

        # Process file types after successful upload
        if file_type in [".pdf", ".jpg", ".jpeg", ".png"]:
            process_pdf_to_text(file_name)
            return {"message": "PDF/Image processed successfully"}, 200
        else:
            return {"error": "Unsupported file type"}, 400

    except NoCredentialsError:
        print("Credentials not available")
        return {"error": "Credentials not available"}, 500

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": f"An error occurred: {e}"}, 500


# functions for processing the pdf file data to text file data
def process_textract_response(response_pages):
    # Initialize a list to hold all extracted text
    all_text = []

    # Iterate over each page of Textract results
    for page in response_pages:
        # Check and extract text from each 'LINE' block
        for item in page:
            if item["BlockType"] == "LINE":
                all_text.append(item["Text"])

    # Combine all text into a single string
    return "\n".join(all_text)


def analyze_document(document_name):

    try:
        response = textract_client.start_document_text_detection(
            DocumentLocation={
                "S3Object": {
                    "Bucket": quest_bucket,
                    "Name": document_name,
                }
            }
        )
        print(response)
        # return response
    except Exception as e:
        return f"File {document_name} not found in {quest_bucket}."

    print(document_name)
    print(quest_bucket)

    # Get the JobId from the response
    job_id = response["JobId"]

    # Initialize variables for loop
    status = ""
    pages = []
    while True:

        status_response = textract_client.get_document_text_detection(JobId=job_id)

        status = status_response["JobStatus"]

        print(f"Job status: {status}")

        if status in ["SUCCEEDED", "FAILED"]:
            # When job succeeds, start processing the results
            if "Blocks" in status_response:
                pages.append(status_response["Blocks"])

            next_token = status_response.get("NextToken", None)

            # Handle pagination if there are more pages of results
            while next_token:
                status_response = textract_client.get_document_text_detection(
                    JobId=job_id, NextToken=next_token
                )
                if "Blocks" in status_response:
                    pages.append(status_response["Blocks"])
                next_token = status_response.get("NextToken", None)
            break
        else:
            # Wait a bit before checking the status again
            time.sleep(5)

    if status == "SUCCEEDED":
        processed_response = process_textract_response(pages)
        return processed_response
    else:
        return jsonify({"error": "Text detection failed"}), 500


def process_pdf_to_text(file_name):
    try:
        # Check if the file type is supported
        supported_file_types = {".pdf", ".jpg", ".jpeg", ".png"}
        filename = secure_filename(file_name.filename)
        # modified_filename = filename.replace("_", "")
        # print(modified_filename)
        _, file_extension = os.path.splitext(filename.lower())
        try:
            s3.head_object(Bucket=quest_bucket, Key=filename)
        except Exception as e:
            return f"File {filename} not found in {quest_bucket}."

        if file_extension in supported_file_types:
            # Call analyze_document to get the text content
            processed_response = analyze_document(filename)

            # Use BytesIO as an in-memory buffer
            in_memory_buffer = io.BytesIO()
            in_memory_buffer.write(processed_response.encode("utf-8"))

            # Move the buffer position to the beginning
            in_memory_buffer.seek(0)

            # Upload the in-memory buffer to textract-bucket
            try:
                s3_key_result = f"{filename.rsplit('.', 1)[0]}.txt"

                s3.upload_fileobj(in_memory_buffer, textract_bucket, s3_key_result)

                return {
                    "message": "Text extraction succeeded",
                    "s3_key_result": s3_key_result,
                }
            except Exception as e:
                print(
                    f"An error occurred while uploading the output file to textract-bucket: {e}"
                )
                return {
                    "error": f"An error occurred while uploading the output file to textract-bucket: {e}"
                }, 500
        else:
            return {
                "error": "Unsupported file type. Supported types: .pdf, .jpg, .jpeg, .png"
            }, 400
    except NoCredentialsError:
        print("Credentials not available")
        return {"error": "Credentials not available"}, 500
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": f"An error occurred: {e}"}, 500


def create_prompt(text_input):
    # Customize the prompt based on the input or requirements
    return f"\n\nHuman: Please provide a correct answer and detailed explanation about the following topic: {text_input}"


def ask_question_to_sonnet(question, context, chat_id, session_id):
    prompt = create_prompt(question)
    print("Preparing request for Sonnet")
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 90000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Based on the following text:\n\n{context}\n\nQuestion: {question}",
                        },
                        {"type": "text", "text": f"Prompt: {prompt}"},
                    ],
                }
            ],
        }
    )
    modelId = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    accept = "application/json"
    contentType = "application/json"

    try:
        response = brt.invoke_model(
            body=body, modelId=modelId, accept=accept, contentType=contentType
        )
        print("Received response from Sonnet")
        response_body = json.loads(response.get("body").read())

        if "content" in response_body:
            response_content = response_body["content"]
            generated_answer = " ".join(
                [item["text"] for item in response_content if item["type"] == "text"]
            )
            print(f"Generated Answer: {generated_answer}")

            message = {"UserMessage": question, "BotResponse": generated_answer}

            # Check if the chat already exists
            try:
                existing_item = table.get_item(Key={"ChatID": chat_id})
                if "Item" in existing_item:
                    existing_messages = existing_item["Item"].get("Messages", [])
                    existing_messages.append(message)

                    table.update_item(
                        Key={"ChatID": chat_id},
                        UpdateExpression="SET Messages = :msg, #ts = :ts",
                        ExpressionAttributeNames={"#ts": "Timestamp"},
                        ExpressionAttributeValues={
                            ":msg": existing_messages,
                            ":ts": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                else:
                    existing_messages = [message]
                    table.put_item(
                        Item={
                            "ChatID": chat_id,
                            "UserID": "12345",
                            "SessionID": session_id,
                            "Timestamp": datetime.now(timezone.utc).isoformat(),
                            "Messages": existing_messages,
                        }
                    )

                return {
                    "ChatID": chat_id,
                    "SessionID": session_id,
                    "UserID": "12345",
                    "Messages": existing_messages,
                }
            except Exception as e:
                print(f"Error interacting with DynamoDB: {e}")
                raise
        else:
            return None

    except Exception as e:
        print(f"Error invoking Sonnet model: {e}")
        raise


def list_files_in_quest_bucket():
    try:
        # List objects in the bchat bucket
        response = s3.list_objects_v2(Bucket="bchat")
        if "Contents" in response:
            # Extract the file names from the response
            file_names = [obj["Key"] for obj in response["Contents"]]
            return {"file_names": file_names}, 200
        else:
            return {"message": "No files found in the bchat bucket"}, 404
    except ClientError as e:
        print("Error listing files in the bchat bucket: %s" % e)
        return {"error": "Failed to list files"}, 500


# function to delete the file from bucket
def delete_file_from_s3(file_key):
    response = s3.list_objects_v2(Bucket=quest_bucket, Prefix=file_key)
    if "Contents" not in response:
        return {"message": f"File {file_key} not found in bucket"}, 404
    # If the file exists, delete it
    s3.delete_object(Bucket=quest_bucket, Key=file_key)
    return {"message": f"File {file_key} deleted successfully"}, 200


def generate_question_with_options(bot_responses):
    content = "\n".join(bot_responses)

    prompt = (
        "Based on the following content, generate 5 multiple-choice questions with 3 options each. "
        "For each question, clearly indicate the correct answer using the format: "
        "'Correct answer: [answer]'.\n\n"
        f"Content: {content}\n\n"
        "Questions, options, and correct answers:"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=3000,
        temperature=0.7,
    )

    question_with_options = response.choices[0].message["content"].strip()
    questions_list = []
    lines = question_with_options.split("\n")
    current_question = None
    correct_answer = None

    for line in lines:
        line = line.strip()

        if line.startswith("Correct answer:"):
            correct_answer = line.split("Correct answer:", 1)[1].strip()

        elif line and line[0].isdigit() and line[1] == ".":
            if current_question:
                current_question["correct_answer"] = correct_answer
                questions_list.append(current_question)
            current_question = {
                "question": line.split(". ", 1)[1],
                "options": [],
                "correct_answer": None,
            }
        elif line and current_question is not None:
            current_question["options"].append(line)

    if current_question:
        current_question["correct_answer"] = correct_answer
        questions_list.append(current_question)

    for question in questions_list:
        if question["options"]:
            # Ensure options are unique
            unique_options = list(set(question["options"]))

            # Limit to 3 options if there are enough unique options
            if len(unique_options) > 3:
                unique_options = unique_options[:3]

            # Shuffle the options
            random.shuffle(unique_options)

            # Ensure the correct answer is among the options
            if question["correct_answer"] not in unique_options:
                unique_options[0] = question["correct_answer"]
                random.shuffle(unique_options)

            question["options"] = unique_options

    return json.dumps(questions_list)


def get_bot_responses(chat_id):
    try:
        # Query DynamoDB for the chat item
        response = table.get_item(Key={"ChatID": chat_id})
        item = response.get("Item", None)

        if item:
            # Extract and return the bot responses
            messages = item.get("Messages", [])
            bot_responses = [
                msg.get("BotResponse") for msg in messages if "BotResponse" in msg
            ]
            return {
                "BotResponses": bot_responses
            }  # Return a dictionary instead of jsonify
        else:
            return {"error": "Chat history not found"}, 404

    except Exception as e:
        return {"error": str(e)}, 500


def get_user_questions(chat_id):
    try:
        # Query DynamoDB for the chat item
        response = table.get_item(Key={"ChatID": chat_id})
        item = response.get("Item", None)

        if item:
            # Extract and return the bot responses
            messages = item.get("Messages", [])
            user_question = [
                msg.get("UserMessage") for msg in messages if "UserMessage" in msg
            ]
            return {
                "UserMessage": user_question
            }  # Return a dictionary instead of jsonify
        else:
            return {"error": "Chat history not found"}, 404

    except Exception as e:
        return {"error": str(e)}, 500


def extract_education_entities(text):
    prompt = (
        f"Extract only educational concepts and entities from the following text. Exclude any actions or non-educational terms.\n\n"
        f"Text: {text}\n\n"
        "Education-related Entities:"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        temperature=0.5,
    )

    entities = response.choices[0].message["content"].strip()
    return entities


# ===========================================================================================================


# =====================================================================================
@app.route("/dev/get_bot_responses", methods=["GET"])
def fetch_bot_responses():
    chat_id = request.form.get("chat_id")
    if not chat_id:
        return jsonify({"error": "Chat ID is required"}), 400

    return get_user_questions(chat_id)


@app.route("/dev/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        response, status_code = upload_to_s3(file)
        return jsonify(response), status_code

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# route of processing the pdf file to text file
@app.route("/dev/pdf-text", methods=["POST"])
def extract_text():
    try:
        content = request.json
        document_name = content["document_name"]

        # Call the process_pdf_to_text function
        result = process_pdf_to_text(document_name)

        # Return the result as JSON
        return jsonify(result)

    except KeyError as ke:
        return jsonify({"error": f"KeyError: {str(ke)}"}), 400

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# route of bedrock


@app.route("/dev/bedrock_newchat", methods=["POST"])
def ask_bedrock_newchat():
    try:
        generated_answer = ""

        if request.method == "POST":
            question = request.form.get("search")
            file_name = request.form.get("file")

            if not question:
                return jsonify({"error": "Please enter your question"}), 400

            if not file_name:
                return jsonify({"error": "Please select pdf or excel only"}), 400

            # Create a new session ID and chat ID for the new chat

            chat_id = str(uuid.uuid4())
            session_id = str(uuid.uuid4())

            file_type = get_file_type(file_name)[1:]
            if file_type in ["pdf", "png", "jpeg"]:
                try:
                    s3_key_result = f"{file_name.rsplit('.', 1)[0]}.txt"
                    print(s3_key_result)
                    try:
                        # Use head_object to check if the object exists
                        s3.head_object(Bucket=textract_bucket, Key=s3_key_result)
                        # If it exists, call get_pdfs
                        content = get_pdfs(s3_key_result)
                    except ClientError as e:
                        if e.response["Error"]["Code"] == "404":
                            # If not found, call process_pdf_to_text
                            content = process_pdf_to_text(file_name)
                        else:
                            # Re-raise the exception if it's not a 404 error
                            raise e

                except Exception as s3_error:
                    return (
                        jsonify({"error": f"An error occurred with S3: {s3_error}"}),
                        500,
                    )
            else:
                return (
                    jsonify(
                        {"error": "Please select a valid file type (pdf, png, jpeg)"}
                    ),
                    400,
                )

            print(question)
            generated_answer = ask_question_to_sonnet(
                question, content, session_id, chat_id
            )

            # Return the generated answer as a JSON response
            return jsonify({"answer": generated_answer}), 200
        else:
            return jsonify({"error": "Wrong method!"}), 405

    except Exception as e:
        print(e)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/dev/ask-bedrock", methods=["POST"])
def ask_bedrock():
    try:
        if request.method == "POST":
            question = request.form.get("search")
            file_name = request.form.get("file")
            chat_id = request.form.get("chat_id")
            session_id = request.form.get("session_id")

            if not question:
                return jsonify({"error": "Please enter your question"}), 400

            if not file_name:
                return jsonify({"error": "Please select pdf or excel only"}), 400

            if not session_id or not chat_id:
                return (
                    jsonify(
                        {
                            "error": "session_id and chat_id are required for continuing a chat."
                        }
                    ),
                    400,
                )

            file_type = get_file_type(file_name)[1:]
            if file_type in ["pdf", "png", "jpeg"]:
                try:
                    s3_key_result = f"{file_name.rsplit('.', 1)[0]}.txt"
                    print(f"S3 Key: {s3_key_result}")
                    try:
                        s3.head_object(Bucket=textract_bucket, Key=s3_key_result)
                        content = get_pdfs(s3_key_result)
                    except ClientError as e:
                        if e.response["Error"]["Code"] == "404":
                            content = process_pdf_to_text(file_name)
                        else:
                            raise e

                except Exception as s3_error:
                    return (
                        jsonify({"error": f"An error occurred with S3: {s3_error}"}),
                        500,
                    )
            else:
                return (
                    jsonify(
                        {"error": "Please select a valid file type (pdf, png, jpeg)"}
                    ),
                    400,
                )

            print(f"Question: {question}")
            generated_answer = ask_question_to_sonnet(
                question, content, chat_id, session_id
            )

            return jsonify({"answer": generated_answer}), 200
        else:
            return jsonify({"error": "Wrong method!"}), 405

    except Exception as e:
        print(f"Error in /dev/ask-bedrock route: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/dev/list", methods=["POST"])
def list_files():
    try:
        current_app.logger.info(f"Using S3 bucket:{quest_bucket}")
        result = list_files_in_quest_bucket()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500


# route function to start quiz
@app.route("/dev/start_quiz", methods=["POST"])
def start_quiz():
    data = request.json
    session_id = data.get("session_id")
    chat_id = data.get("chat_id")

    if not session_id or not chat_id:
        return jsonify({"error": "Session ID and chat ID are required"}), 400

    # Fetch bot responses using the get_bot_responses function
    bot_responses_response = get_bot_responses(chat_id)

    # Check if the response is an error
    if isinstance(bot_responses_response, dict) and "error" in bot_responses_response:
        return jsonify(bot_responses_response), bot_responses_response.get(
            "status_code", 500
        )

    bot_responses_data = bot_responses_response
    bot_responses = bot_responses_data.get("BotResponses", [])
    print("DEBUG :: bot_responses :", bot_responses)

    if not bot_responses:
        return jsonify({"error": "No bot responses found for the given chat ID"}), 404

    # Generate questions based on bot responses
    questions_list = generate_question_with_options(bot_responses)
    print("DEBUG :: mcq question :", questions_list)
    if isinstance(questions_list, str):
        try:
            questions_list = json.loads(questions_list)
        except json.JSONDecodeError:
            return jsonify({"error": "Failed to parse questions."}), 500

    if not questions_list or not isinstance(questions_list, list):
        return jsonify({"error": "No questions generated or incorrect format."}), 500

    # Store questions in in-memory storage
    questions_store["questions"] = questions_list
    score_store["score"] = 0

    # Debugging: Check stored questions
    print("Stored questions:", questions_store["questions"])

    return jsonify(
        {
            "message": "Quiz started!",
            "questions": questions_list,
        }
    )


# route function to submit answers of quiz
@app.route("/submit_answers", methods=["POST"])
def submit_answers():
    data = request.json
    answers = data.get("answers")
    questions = questions_store.get("questions", [])

    # Debugging: Check questions and answers
    print("Questions from store:", questions)
    print("Submitted answers:", answers)

    if not questions or len(answers) != len(questions):
        return jsonify({"error": "Answers do not match the number of questions."}), 400

    score = 0
    detailed_results = []

    # Convert the keys to integers and validate
    try:
        converted_answers = {int(key): value.strip() for key, value in answers.items()}
    except ValueError as e:
        return jsonify({"error": f"Invalid answer key: {str(e)}"}), 400

    for index in range(1, len(questions) + 1):
        user_answer = converted_answers.get(index, "")

        # Check if the user's answer matches the correct answer
        if user_answer == questions[index - 1]["correct_answer"]:
            score += 1
            is_correct = True
        else:
            is_correct = False

        # Store detailed result
        detailed_results.append(
            {
                "question": questions[index - 1]["question"],
                "selected_answer": user_answer,
                "correct_answer": questions[index - 1]["correct_answer"],
                "is_correct": is_correct,
            }
        )

    # Update score in in-memory storage
    score_store["score"] = score

    return jsonify(
        {
            "message": "Quiz completed!",
            "score": score,
            "total_questions": len(questions),
            "details": detailed_results,  # Return detailed results
        }
    )


def list_files_in_s3():

    try:
        # List objects within the specified bucket
        response = s3.list_objects_v2(Bucket=quest_bucket)

        # Check if the bucket is empty
        if "Contents" in response:
            print("Files in the bucket:")
            for obj in response["Contents"]:
                print(obj["Key"])  # 'Key' is the file name
        else:
            print("No files found in the bucket.")

    except Exception as e:
        print(f"An error occurred: {e}")


@app.route("/dev/list-files-s3", methods=["GET"])
def list_s3_files():
    # Replace with your bucket name
    files = list_files_in_s3()

    if isinstance(files, str):  # If an error occurred, files will be an error string
        return jsonify({"error": files}), 500

    return jsonify({"files": files}), 200


@app.route("/dev/delete_file", methods=["DELETE"])
def delete_file():
    data = request.json
    file_key = data.get("file_key")

    if not file_key:
        return jsonify({"error": "Object key not provided"}), 400

    result = delete_file_from_s3(file_key)
    return jsonify(result[0]), result[1]


@app.route("/dev/get-file/<pdf_path>", methods=["GET"])
def read_pdf_route(pdf_path):
    if not pdf_path:
        return jsonify({"error": "PDF path not provided"}), 400

    try:
        pdf_content = read_pdfs(pdf_path)
        encoded_content = base64.b64encode(pdf_content).decode("utf-8")
        return (
            jsonify(
                {
                    "message": "PDF content retrieved successfully",
                    "pdf_content": encoded_content,
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
