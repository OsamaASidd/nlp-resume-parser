import re
import logging
import json
import pdfplumber
import openai
from tokenizer import num_tokens_from_string
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

class OptimizedResumeParser():
    def __init__(self, OPENAI_API_KEY):
        # Set GPT API key
        openai.api_key = OPENAI_API_KEY

        # Optimized, more concise prompt
        self.prompt_questions = \
"""Extract resume data as JSON:
{
  "basic_info": {
    "first_name": "", "last_name": "", "full_name": "", "email": "", 
    "phone_number": "", "location": "", "portfolio_website_url": "", 
    "linkedin_url": "", "github_main_page_url": "", "university": "", 
    "education_level": "", "graduation_year": "", "graduation_month": "", 
    "majors": [], "GPA": ""
  },
  "work_experience": [{"job_title": "", "company": "", "location": "", "duration": "", "job_summary": ""}],
  "project_experience": [{"project_name": "", "project_description": ""}]
}

Resume text:"""

        # Set up logger
        logging.basicConfig(filename='logs/parser.log', level=logging.DEBUG)
        self.logger = logging.getLogger()

    def pdf2string_optimized(self, pdf_path: str) -> str:
        """
        Optimized PDF extraction with better text cleaning and page limits.
        """
        pdf_str = ""
        page_count = 0
        max_pages = 3  # Limit processing to first 3 pages for speed
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                if page_count >= max_pages:
                    break
                    
                # Extract text with better formatting preservation
                page_text = page.extract_text()
                if page_text:
                    pdf_str += page_text + "\n\n"
                page_count += 1

        # More efficient text cleaning
        pdf_str = re.sub(r'\n{3,}', '\n\n', pdf_str)  # Reduce excessive newlines
        pdf_str = re.sub(r'[ \t]{2,}', ' ', pdf_str)  # Reduce excessive spaces
        pdf_str = re.sub(r'http[s]?://', '', pdf_str)  # Remove protocol prefixes
        
        # Limit text length to control token usage
        max_chars = 4000  # Roughly 1000 tokens
        if len(pdf_str) > max_chars:
            pdf_str = pdf_str[:max_chars] + "\n[Text truncated for processing efficiency]"
            
        return pdf_str.strip()

    def query_completion_optimized(self,
                                 prompt: str,
                                 engine: str = 'gpt-3.5-turbo',
                                 temperature: float = 0.0,
                                 max_tokens: int = 800,  # Reduced from 1000
                                 top_p: float = 0.9,  # Slightly more focused
                                 frequency_penalty: float = 0,
                                 presence_penalty: float = 0) -> str:
        """
        Optimized completion with faster model and better parameters.
        """
        self.logger.info(f'query_completion: using {engine}')

        # Use optimized parameters for faster response
        messages = [
            {"role": "system", "content": "You are a fast, accurate resume parser. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                timeout=15  # Add timeout to prevent hanging
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            return "{}"  # Return empty JSON on error

    def preprocess_resume_text(self, text: str) -> str:
        """
        Smart preprocessing to extract only relevant sections quickly.
        """
        # Define sections to prioritize
        important_sections = [
            'education', 'experience', 'work', 'employment', 'projects', 
            'skills', 'contact', 'email', 'phone', 'address'
        ]
        
        lines = text.split('\n')
        relevant_lines = []
        current_section_important = False
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if line contains contact info (always important)
            if any(keyword in line_lower for keyword in ['@', 'phone', 'email', 'linkedin', 'github']):
                relevant_lines.append(line)
                continue
                
            # Check if line is a section header
            if any(section in line_lower for section in important_sections):
                current_section_important = True
                relevant_lines.append(line)
                continue
                
            # If we're in an important section, keep the line
            if current_section_important and line.strip():
                relevant_lines.append(line)
                
            # Reset section importance on empty line (potential section break)
            if not line.strip():
                current_section_important = False
                
        return '\n'.join(relevant_lines)

    def query_resume_fast(self, pdf_path: str) -> dict:
        """
        Optimized resume parsing with multiple speed improvements.
        """
        try:
            # Step 1: Fast PDF extraction
            pdf_str = self.pdf2string_optimized(pdf_path)
            
            # Step 2: Smart preprocessing to reduce token count
            processed_text = self.preprocess_resume_text(pdf_str)
            
            # Step 3: Create optimized prompt
            prompt = self.prompt_questions + '\n' + processed_text
            
            # Log token usage for monitoring
            estimated_tokens = num_tokens_from_string(prompt, 'gpt-3.5-turbo')
            self.logger.info(f'Estimated prompt tokens: {estimated_tokens}')
            
            # Step 4: Use faster model with optimized parameters
            engine = 'gpt-3.5-turbo'  # Faster than gpt-4
            max_tokens = min(800, 4000 - estimated_tokens)  # Adaptive token limit
            
            response_text = self.query_completion_optimized(
                prompt, 
                engine=engine, 
                max_tokens=max_tokens
            )
            
            # Step 5: Parse JSON with error handling
            try:
                # Clean response text before parsing
                response_text = response_text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3]
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3]
                
                resume = json.loads(response_text)
                
                # Validate basic structure
                if 'basic_info' not in resume:
                    resume['basic_info'] = {}
                if 'work_experience' not in resume:
                    resume['work_experience'] = []
                if 'project_experience' not in resume:
                    resume['project_experience'] = []
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON: {str(e)}")
                self.logger.error(f"Response text: {response_text}")
                # Return minimal structure on parse error
                resume = {
                    "basic_info": {},
                    "work_experience": [],
                    "project_experience": []
                }
            
            return resume
            
        except Exception as e:
            self.logger.error(f"Error in query_resume_fast: {str(e)}")
            return {
                "basic_info": {},
                "work_experience": [],
                "project_experience": [],
                "error": str(e)
            }

    # Keep original method for backward compatibility
    def query_resume(self, pdf_path: str) -> dict:
        """
        Original method - kept for backward compatibility.
        Use query_resume_fast() for better performance.
        """
        return self.query_resume_fast(pdf_path)