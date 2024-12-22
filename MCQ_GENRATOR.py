import os
import json
import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.callbacks import get_openai_callback
import traceback
import re
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load environment variables
load_dotenv()
KEY = os.getenv("Groq_key")


# Initialize the language model
llm = ChatGroq(groq_api_key=KEY, model_name="mixtral-8x7b-32768", temperature=0)


RESPONSE_JSON = {
    "1": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "2": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "3": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
}

TEMPLATE = """
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to 
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. 
Ensure to make {number} MCQs, please create a mcq which are provided in json format.
### RESPONSE_JSON
{RESPONSE_JSON}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "RESPONSE_JSON"],
    template=TEMPLATE
)


quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)

TEMPLATE2 = """
You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students. 
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
If the quiz is not at par with the cognitive and analytical abilities of the students, 
update the quiz questions which need to be changed and change the tone such that it perfectly fits the student abilities.
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(input_variables=["subject", "quiz"], template=TEMPLATE2)


review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)


generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain], 
    input_variables=["text", "number", "subject", "tone", "RESPONSE_JSON"], 
    output_variables=["quiz", "review"], 
    verbose=True
)

# Function to extract JSON from the response
def extract_json_from_response(response_text):
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        return match.group(0)
    return None

# Function to create a PDF from the quiz dictionary
def create_pdf(quiz_dict):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 50, "Generated MCQs")
    y_position = height - 70

    c.setFont("Helvetica", 12)
    for i, (key, value) in enumerate(quiz_dict.items(), 1):
        if y_position < 100:
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = height - 70
        c.drawString(100, y_position, f"Q{i}: {value['mcq']}")
        y_position -= 20
        for option_key, option_value in value['options'].items():
            c.drawString(120, y_position, f"- {option_key.upper()}: {option_value}")
            y_position -= 20
        c.drawString(100, y_position, f"Correct Answer: {value['correct']}")
        y_position -= 30

    c.save()
    buffer.seek(0)
    return buffer

# Streamlit UI
st.title("MCQ Generator")

subject = st.text_input("Subject")
number = st.number_input("Number of MCQs", min_value=1, step=1)
tone = st.selectbox("Tone", ["simple", "medium", "hard"])
file = st.file_uploader("Upload a PDF or Text file", type=["pdf", "txt"])
generate_button = st.button("Generate MCQs")

if generate_button:
    if subject and number and tone and file is not None:
        try:
            # Extract text from the uploaded file
            if file.type == "application/pdf":
                reader = PyPDF2.PdfReader(file)
                TEXT = ""
                for page in range(len(reader.pages)):
                    TEXT += reader.pages[page].extract_text()
            elif file.type == "text/plain":
                TEXT = file.read().decode("utf-8")

            with st.spinner("Generating and evaluating MCQs..."):
                with get_openai_callback() as cb:
                    response = generate_evaluate_chain(
                        {
                            "text": TEXT,
                            "number": number,
                            "subject": subject,
                            "tone": tone,
                            "RESPONSE_JSON": json.dumps(RESPONSE_JSON)
                        }
                    )

                    # Extract quiz and review outputs
                    quiz = response["quiz"]
                    review = response["review"]

                    # Extract JSON from the quiz
                    quiz_json = extract_json_from_response(quiz)
                    if quiz_json:
                        try:
                            quiz_dict = json.loads(quiz_json)
                            pdf_buffer = create_pdf(quiz_dict)
                            st.download_button(
                                label="Download MCQs as PDF",
                                data=pdf_buffer,
                                file_name="MCQs.pdf",
                                mime="application/pdf",
                            )
                        except json.JSONDecodeError as e:
                            st.error(f"Error parsing quiz JSON: {e}")
                            st.write("Quiz text:", quiz)
                    else:
                        st.error("Could not extract valid JSON from the quiz response.")
                        st.write("Quiz text:", quiz)
        except Exception as e:
            st.error(f"Error processing the file: {e}")
            st.error(traceback.format_exc())
    else:
        st.error("Please fill in all fields and upload a file.")
