import os
import secrets
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from env_parser import parse_env_file
from parser import OptimizedResumeParser
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf'}

env_variables = parse_env_file()

app = Flask(__name__)
CORS(app) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.add_url_rule("/resume/<name>", endpoint="resume", build_only=True)
app.secret_key = secrets.token_urlsafe(32)

parser = OptimizedResumeParser(os.getenv('OPENAI_API_KEY', env_variables.get('OPENAI_API_KEY')))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# New API endpoint for direct PDF upload and parsing
@app.route("/api/parse-resume", methods=['POST'])
def parse_resume_api():
    try:
        # Check if PDF file is in the request
        if 'pdf' not in request.files:
            return jsonify({
                'error': 'No PDF file provided',
                'message': 'Please include a PDF file with key "pdf" in the request'
            }), 400
        
        file = request.files['pdf']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select a PDF file to upload'
            }), 400
        
        # Check if file is PDF
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'message': 'Only PDF files are allowed'
            }), 400
        
        # Save the file temporarily
        filename = secure_filename(file.filename)
        # Add timestamp to avoid filename conflicts
        import time
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Parse the resume
        resume_data = parser.query_resume(file_path)
        
        # Clean up - remove the temporary file
        try:
            os.remove(file_path)
        except OSError:
            pass  # File removal failed, but continue
        
        # Return the parsed resume data
        return jsonify({
            'success': True,
            'data': resume_data
        }), 200
        
    except Exception as e:
        # Clean up file on error
        try:
            if 'file_path' in locals():
                os.remove(file_path)
        except:
            pass
            
        return jsonify({
            'error': 'Processing failed',
            'message': str(e)
        }), 500

# Keep original routes for backward compatibility
@app.route("/", methods=['GET', 'POST'])
@app.route("/resume", methods=['GET', 'POST'])
def upload_resume():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('resume', name=filename))
    return render_template('index.html')

@app.route('/resume/<name>')
def display_resume(name):
    resume_path = os.path.join(app.config["UPLOAD_FOLDER"], name)
    return parser.query_resume(resume_path)

# Health check endpoint
@app.route("/api/health", methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'resume-parser-service'
    }), 200

if __name__ == "__main__":
    host = os.getenv("RESUME_PARSER_HOST", '127.0.0.1')
    port = os.getenv("RESUME_PARSER_PORT", '5000')
    assert port.isnumeric(), 'port must be an integer'
    port = int(port)
    app.run(host=host, port=port, debug=True)