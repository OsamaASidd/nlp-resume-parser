import re
import logging
import json
import pdfplumber  # Using pdfplumber instead of pdftotext
import openai
from tokenizer import num_tokens_from_string

class ResumeParser():
    def __init__(self, OPENAI_API_KEY):
        # Set GPT API key
        openai.api_key = OPENAI_API_KEY

        # GPT prompt
        self.prompt_questions = \
"""Summarize the text below into a JSON with exactly the following structure {basic_info: {first_name, last_name, full_name, email, phone_number, location, portfolio_website_url, linkedin_url, github_main_page_url, university, education_level (BS, MS, or PhD), graduation_year, graduation_month, majors, GPA}, work_experience: [{job_title, company, location, duration, job_summary}], project_experience:[{project_name, project_description}]}
"""
        # Set up logger
        logging.basicConfig(filename='logs/parser.log', level=logging.DEBUG)
        self.logger = logging.getLogger()

    def pdf2string(self, pdf_path: str) -> str:
        """
        Extract the content of a pdf file to string using pdfplumber.
        """
        pdf_str = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                pdf_str += page.extract_text() + "\n\n"

        # Clean up the text
        pdf_str = re.sub(r'\s[,.]', ',', pdf_str)
        pdf_str = re.sub(r'[\n]+', '\n', pdf_str)
        pdf_str = re.sub(r'[\s]+', ' ', pdf_str)
        pdf_str = re.sub(r'http[s]?(://)?', '', pdf_str)
        return pdf_str

    def query_completion(self,
                         prompt: str,
                         engine: str = 'gpt-3.5-turbo',
                         temperature: float = 0.0,
                         max_tokens: int = 1000,
                         top_p: int = 1,
                         frequency_penalty: int = 0,
                         presence_penalty: int = 0) -> str:
        """
        Handle both chat and legacy completion models.
        """
        self.logger.info(f'query_completion: using {engine}')

        if engine.startswith("gpt-3.5") or engine.startswith("gpt-4"):
            messages = [
                {"role": "system", "content": "You are a resume parsing assistant."},
                {"role": "user", "content": prompt}
            ]
            response = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            return response['choices'][0]['message']['content']
        else:
            estimated_prompt_tokens = num_tokens_from_string(prompt, engine)
            estimated_answer_tokens = max_tokens - estimated_prompt_tokens
            self.logger.info(f'Tokens: {estimated_prompt_tokens} + {estimated_answer_tokens} = {max_tokens}')

            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temperature,
                max_tokens=estimated_answer_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            return response['choices'][0]['text']

    def query_resume(self, pdf_path: str) -> dict:
        """
        Query GPT for the resume summary from the PDF.
        """
        pdf_str = self.pdf2string(pdf_path)
        print(pdf_str)
        prompt = self.prompt_questions + '\n' + pdf_str

        engine = 'gpt-3.5-turbo'
        max_tokens = 3000  # Adjust based on actual expected response

        response_text = self.query_completion(prompt, engine=engine, max_tokens=max_tokens)
        print(response_text)

        try:
            resume = json.loads(response_text)
        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse JSON from response: " + str(e))
            resume = {}

        return resume
