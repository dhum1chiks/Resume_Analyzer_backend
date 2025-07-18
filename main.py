import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from pdfminer.high_level import extract_text_to_fp
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import logging
import json
from io import BytesIO
from starlette.concurrency import run_in_threadpool
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, List
import uuid
import re
import time
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Resume Analysis API")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:3001",
        "https://resume-analyzer-frontend-m03wwwh7e-manis-projects-f441ec6f.vercel.app",
        "https://resume-analyzer-frontend-ten.vercel.app",
        "https://resume-analyzer-backend-nine.vercel.app",
        "https://resume-analyzer-backend-7i5awzgfw-manis-projects-f441ec6f.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set Supabase credentials
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("SUPABASE_URL or SUPABASE_KEY environment variable not set")
    print("Error: SUPABASE_URL and SUPABASE_KEY environment variables are required. Check your .env file.")
    exit(1)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Set Groq API key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable not set")
    print("Error: GROQ_API_KEY environment variable is required. Check your .env file.")
    exit(1)
client = Groq(api_key=GROQ_API_KEY)

# Initialize Supabase table
def init_db():
    try:
        # Check if the table exists by attempting to select from it
        response = supabase.table('tailoring_attempts').select('id').limit(1).execute()
        logger.debug("Table 'tailoring_attempts' exists")
        
        # Check if target_role column exists
        schema_response = supabase.from_('information_schema.columns')\
            .select('column_name')\
            .eq('table_name', 'tailoring_attempts')\
            .eq('column_name', 'target_role')\
            .execute()
        
        if not schema_response.data:
            logger.warning("target_role column missing. Please add it manually with:")
            logger.warning("""
                ALTER TABLE tailoring_attempts
                ADD COLUMN IF NOT EXISTS target_role TEXT
            """)
    except Exception as e:
        logger.error(f"Table check failed: {str(e)}")
        logger.warning("Please create the 'tailoring_attempts' table manually with the following SQL:")
        logger.warning("""
            CREATE TABLE tailoring_attempts (
                id UUID PRIMARY KEY,
                user_id TEXT,
                resume TEXT,
                job_description TEXT,
                analysis_result JSONB,
                cover_letter TEXT,
                template_id TEXT,
                tone TEXT,
                target_role TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        logger.warning("Run this in the Supabase SQL Editor and redeploy the application.")

init_db()

# Serve favicon to prevent 404 errors
@app.get("/favicon.ico")
async def favicon():
    raise HTTPException(status_code=404, detail="Favicon not found")

@app.get("/favicon.png")
async def favicon_png():
    raise HTTPException(status_code=404, detail="Favicon not found")

# Helper function to clean potential JSON response
def clean_json_response(raw_response: str) -> str:
    if not raw_response:
        return '{"error": "Empty response from AI model"}'
    raw_response = re.sub(r'```json\n|```', '', raw_response).strip()
    if not raw_response.startswith('{') and not raw_response.endswith('}'):
        raw_response = '{' + raw_response + '}'
    return raw_response

# Helper function to interact with Groq with retry mechanism
def ask(message, sys_message="You are a senior developer.", model="llama3-8b-8192", max_retries=3):
    messages = [
        {
            "role": "system",
            "content": sys_message + """
            Always respond with a valid JSON string. Ensure proper JSON formatting with double quotes for keys and values.
            If an error occurs, return a JSON object with an 'error' key containing the error message.
            Example: {"skills": [], "match_percentage": 0, "suggestions": [], "error": null}
            """
        },
        {"role": "user", "content": message}
    ]
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Sending Groq request, attempt {attempt + 1}")
            response = client.chat.completions.create(model=model, messages=messages)
            raw_response = response.choices[0].message.content
            logger.debug(f"Raw Groq response: {raw_response}")
            cleaned_response = clean_json_response(raw_response)
            try:
                parsed_response = json.loads(cleaned_response)
                if not isinstance(parsed_response, dict):
                    logger.error(f"Response is not a JSON object: {cleaned_response}")
                    return {"error": "Response is not a valid JSON object"}
                if "error" not in parsed_response:
                    parsed_response["error"] = None
                return parsed_response
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error on attempt {attempt + 1}: {str(e)}")
                logger.debug(f"Failed response: {cleaned_response}")
                if attempt == max_retries - 1:
                    return {"error": f"Invalid JSON response after {max_retries} attempts: {str(e)}"}
        except Exception as e:
            logger.error(f"Groq API error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return {"error": f"Groq API error after {max_retries} attempts: {str(e)}"}
        time.sleep(1)
    return {"error": "Failed to get valid response after maximum retries"}

# Helper functions for text extraction
async def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        output = BytesIO()
        await run_in_threadpool(extract_text_to_fp, BytesIO(file_content), output)
        text = output.getvalue().decode('utf-8', errors='ignore')
        return text.strip() if text else "No text found in the PDF"
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}")

async def extract_text_from_docx(file_content: bytes) -> str:
    try:
        doc = await run_in_threadpool(Document, BytesIO(file_content))
        text = '\n'.join([para.text for para in doc.paragraphs if para.text])
        return text.strip() if text else "No text found in the DOCX"
    except Exception as e:
        logger.error(f"DOCX extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"DOCX extraction failed: {str(e)}")

# Helper function to generate PDF
def generate_pdf(content: dict, template_id: str) -> BytesIO:
    try:
        logger.debug(f"Generating PDF with template_id: {template_id}, content keys: {list(content.keys())}")
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        y_position = height - 50

        templates = {
            "modern": {"font": "Courier", "size": 12, "color": (0, 0, 0)},
            "classic": {"font": "Courier", "size": 11, "color": (0, 0, 0)},
            "professional": {"font": "Courier", "size": 12, "color": (0, 0, 0.5)}
        }
        
        template = templates.get(template_id, templates["modern"])
        logger.debug(f"Selected template: {template}")
        
        try:
            c.setFont(template["font"], template["size"])
            logger.debug(f"Font set to {template['font']} with size {template['size']}")
        except Exception as e:
            logger.error(f"Font error: {str(e)}, falling back to Helvetica")
            c.setFont("Helvetica", template["size"])
        
        c.setFillColor(template["color"])
        
        MAX_TEXT_LENGTH = 100000
        resume_text = content.get("resume", "")
        if len(resume_text) > MAX_TEXT_LENGTH:
            logger.error("Resume text exceeds maximum length")
            raise ValueError("Resume text exceeds maximum length of 100,000 characters")
        
        c.drawString(50, y_position, "Resume")
        y_position -= 30
        resume_text = resume_text.encode('ascii', 'ignore').decode('ascii')
        for line in resume_text.split("\n"):
            if y_position < 50:
                c.showPage()
                y_position = height - 50
                c.setFont(template["font"], template["size"])
            c.drawString(50, y_position, line[:80])
            y_position -= 15
        
        if content.get("cover_letter"):
            cover_letter_text = content["cover_letter"]
            if len(cover_letter_text) > MAX_TEXT_LENGTH:
                logger.error("Cover letter text exceeds maximum length")
                raise ValueError("Cover letter text exceeds maximum length of 100,000 characters")
            c.showPage()
            y_position = height - 50
            c.drawString(50, y_position, "Cover Letter")
            y_position -= 30
            cover_letter_text = cover_letter_text.encode('ascii', 'ignore').decode('ascii')
            for line in cover_letter_text.split("\n"):
                if y_position < 50:
                    c.showPage()
                    y_position = height - 50
                    c.setFont(template["font"], template["size"])
                c.drawString(50, y_position, line[:80])
                y_position -= 15
        
        c.save()
        buffer.seek(0)
        logger.debug("PDF generation completed successfully")
        return buffer
    except Exception as e:
        logger.error(f"PDF generation error: {str(e)}")
        raise

@app.get("/home")
def home():
    return {"message": "FastAPI on Vercel"}

# Endpoint to extract text from files
@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing file: {file.filename}")
        if not file.filename.lower().endswith(('.pdf', '.docx')):
            raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF or DOCX.")
        MAX_FILE_SIZE = 10 * 1024 * 1024
        file_size = file.size
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")
        file_content = await file.read()
        if file.filename.lower().endswith('.pdf'):
            text = await extract_text_from_pdf(file_content)
        else:
            text = await extract_text_from_docx(file_content)
        logger.info(f"Extracted text length: {len(text)}")
        return {"text": text}
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

# Pydantic model for analysis input
class AnalysisInput(BaseModel):
    resume: str
    job_description: str
    generate_cover_letter: bool = False
    target_role: Optional[str] = None
    tone: Optional[str] = "formal"
    user_id: Optional[str] = None
    template_id: Optional[str] = "modern"
    generate_interview_questions: bool = False

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if len(v["resume"]) > 20000:
            raise ValueError("Resume text exceeds 20,000 characters")
        if len(v["job_description"]) > 20000:
            raise ValueError("Job description text exceeds 20,000 characters")
        if v.get("tone") and v["tone"] not in ["formal", "friendly", "technical", "casual"]:
            raise ValueError("Tone must be 'formal', 'friendly', 'technical', or 'casual'")
        if v.get("template_id") and v["template_id"] not in ["modern", "classic", "professional"]:
            raise ValueError("Template must be 'modern', 'classic', or 'professional'")
        return cls(**v)

# Endpoint to analyze resume and job description
@app.post("/analyze")
async def analyze(input_data: AnalysisInput):
    prompt = f"""
    Analyze the provided resume and job description with a human-like, natural tone, avoiding any indication of AI-generated content. Perform the following tasks:
    1. Extract key skills and requirements from the job description.
    2. Compare the extracted skills with those in the resume, calculate a match percentage (0-100), and identify specific skills from the job description that are missing in the resume.
    3. Provide specific, actionable suggestions to improve the resume based on the job description, focusing on addressing missing skills or enhancing relevance.
    4. Identify minor details or implied questions in the job description (e.g., specific tools, certifications, or soft skills) and provide concise, professional answers or explanations for how the candidate could address them.
    {'5. Consider the target role: ' + input_data.target_role if input_data.target_role else ''}

    Return a valid JSON string with the following keys:
    - "skills": list of skills and requirements extracted from the job description
    - "missing_skills": list of skills from the job description not found in the resume
    - "match_percentage": numeric match percentage (0-100)
    - "suggestions": list of specific, actionable improvements for the resume
    - "minor_details": list of objects with "detail" (the minor detail or question from the job description) and "answer" (a concise, professional response)
    - "error": null if successful, or an error message if applicable
    """
    
    if input_data.generate_cover_letter:
        prompt += f"""
        6. Generate a personalized cover letter based on the resume and job description in a {input_data.tone} tone (formal, friendly, technical, or casual). Ensure the tone is consistent and the content feels human-written. Include the cover_letter as plain text under the "cover_letter" key.
        """
    
    if input_data.generate_interview_questions:
        prompt += """
        7. Generate a list of 10 relevant interview questions based on the job description. Include these under the "interview_questions" key as a list of strings.
        """
    
    prompt += f"""
    Job Description:
    {input_data.job_description}
    
    Resume:
    {input_data.resume}
    
    Ensure the response is a valid JSON string with proper formatting. Example:
    {{
        "skills": ["Python", "JavaScript", "AWS"],
        "missing_skills": ["AWS"],
        "match_percentage": 85,
        "suggestions": ["Add AWS project experience", "Highlight leadership in team projects"],
        "minor_details": [
            {{"detail": "Experience with cloud platforms", "answer": "Highlight any cloud-related projects or consider earning an AWS certification."}},
            {{"detail": "Strong communication skills", "answer": "Emphasize examples of effective communication in your resume."}}
        ],
        "cover_letter": "Dear Hiring Manager, ...",
        "interview_questions": ["How have you used Python in past projects?", "Can you describe your experience with AWS?", "How do you handle tight deadlines?", "What is your approach to teamwork?", "How do you stay updated with technology trends?", "Can you share an example of a challenging project?", "How do you prioritize tasks?", "What tools do you use for collaboration?", "How do you handle feedback?", "What motivates you in this role?"],
        "error": null
    }}
    """

    try:
        analysis = ask(prompt)
        if "error" in analysis and analysis["error"]:
            logger.error(f"Analysis failed: {analysis['error']}")
            raise HTTPException(status_code=500, detail=analysis["error"])
        
        attempt_id = str(uuid.uuid4())
        data = {
            "id": attempt_id,
            "user_id": input_data.user_id or "anonymous",
            "resume": input_data.resume,
            "job_description": input_data.job_description,
            "analysis_result": analysis,
            "cover_letter": analysis.get("cover_letter", ""),
            "template_id": input_data.template_id,
            "tone": input_data.tone,
            "target_role": input_data.target_role,
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        try:
            response = supabase.table('tailoring_attempts').insert(data).execute()
            if response.data is None:
                logger.error(f"Supabase insert failed: {response}")
                raise HTTPException(status_code=500, detail="Failed to save tailoring attempt")
        except Exception as e:
            logger.error(f"Supabase insert error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save tailoring attempt: {str(e)}")
        
        return {"analysis": analysis, "attempt_id": attempt_id}
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logger.error(f"Unexpected error in analyze endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Endpoint to retrieve past tailoring attempts
@app.get("/history/{user_id}")
async def get_history(user_id: str):
    try:
        response = supabase.table('tailoring_attempts').select('*').eq('user_id', user_id).execute()
        logger.debug(f"Supabase response: {response}")
        
        if response.data is None:
            logger.error("Supabase select returned None")
            raise HTTPException(status_code=500, detail="Failed to retrieve history: Invalid response")
        
        attempts = [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "resume": row["resume"],
                "job_description": row["job_description"],
                "analysis_result": row["analysis_result"],
                "cover_letter": row["cover_letter"],
                "template_id": row["template_id"],
                "tone": row["tone"],
                "target_role": row.get("target_role", "Unknown Role")
            } for row in response.data
        ]
        return {"attempts": attempts}
    except Exception as e:
        logger.error(f"History retrieval error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"History retrieval failed: {str(e)}")

# Endpoint to export resume and cover letter as PDF
@app.post("/export-pdf")
async def export_pdf(input_data: AnalysisInput):
    try:
        logger.debug(f"Received PDF export request: template_id={input_data.template_id}, generate_cover_letter={input_data.generate_cover_letter}, resume_length={len(input_data.resume)}, job_desc_length={len(input_data.job_description)}")
        
        if not input_data.resume:
            logger.error("Resume is empty")
            raise HTTPException(status_code=400, detail="Resume text is required")
        if input_data.generate_cover_letter and not input_data.job_description:
            logger.error("Job description missing for cover letter generation")
            raise HTTPException(status_code=400, detail="Job description is required for cover letter generation")
        
        content = {"resume": input_data.resume}
        if input_data.generate_cover_letter:
            logger.debug("Generating cover letter")
            analysis = ask(f"""
                Generate a personalized cover letter based on the resume and job description.
                Use a {input_data.tone} tone.
                Job Description: {input_data.job_description}
                Resume: {input_data.resume}
                Return a JSON object with a "cover_letter" key.
                Example: {{"cover_letter": "Dear Hiring Manager, ...", "error": null}}
            """)
            if "error" in analysis and analysis["error"]:
                logger.error(f"Cover letter generation failed: {analysis['error']}")
                raise HTTPException(status_code=500, detail=analysis["error"])
            content["cover_letter"] = analysis.get("cover_letter", "")
            logger.debug(f"Cover letter generated: {content['cover_letter'][:100]}...")
        
        pdf_buffer = generate_pdf(content, input_data.template_id)
        headers = {
            "Content-Disposition": f'attachment; filename="resume_{input_data.template_id}_{int(time.time())}.pdf"'
        }
        logger.debug("Returning PDF response")
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers=headers
        )
    except Exception as e:
        logger.error(f"PDF export error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF export failed: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
