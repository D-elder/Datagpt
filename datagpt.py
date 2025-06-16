"""
Enhanced Conversational Data Analytics Application
A modern, robust implementation of a DataGPT-like application with:
- Improved architecture and error handling
- Modern responsive UI
- Enhanced visualization capabilities
- Better security practices
- Intelligent SQL generation with safety checks
"""

import os
import json
import sqlite3
import pandas as pd
from flask import Flask, render_template_string, request, jsonify, session
from werkzeug.utils import secure_filename
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyticsApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')
        self.setup_config()
        self.setup_routes()
        
    def setup_config(self):
        """Initialize configuration with environment variables"""
        # Validate required environment variables
        required_vars = ['ANTHROPIC_API_KEY', 'SECRET_KEY']
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            logger.error("Please check your .env file and ensure all required variables are set.")
            
        # Anthropic Configuration
        self.anthropic_client = None
        self.model_name = os.environ.get('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229')
        
        try:
            import anthropic
            self.anthropic_client = anthropic.Anthropic(
                api_key=os.environ.get('ANTHROPIC_API_KEY'),
                base_url=os.environ.get('ANTHROPIC_API_BASE', 'https://api.anthropic.com')
            )
            logger.info("Anthropic Claude client initialized successfully")
        except ImportError:
            logger.error("Anthropic library not installed. Run: pip install anthropic")
            self.anthropic_client = None
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            logger.error("The application will continue but AI features will not work.")
            self.anthropic_client = None
            
        # Flask configuration
        self.app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
        self.app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
        
        # File upload configuration
        self.upload_folder = os.environ.get('UPLOAD_FOLDER', 'uploads')
        self.database_folder = os.environ.get('DATABASE_FOLDER', 'databases')
        
        # Create necessary directories
        os.makedirs(self.upload_folder, exist_ok=True)
        os.makedirs(self.database_folder, exist_ok=True)
        
        # Logging configuration
        log_level = os.environ.get('LOG_LEVEL', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def setup_routes(self):
        """Setup Flask routes"""
        self.app.route('/', methods=['GET'])(self.index)
        self.app.route('/upload', methods=['POST'])(self.upload_file)
        self.app.route('/query', methods=['POST'])(self.handle_query)
        self.app.route('/tables', methods=['GET'])(self.get_tables)
        self.app.route('/reset', methods=['POST'])(self.reset_data)

    def get_database_connection(self) -> sqlite3.Connection:
        """Get database connection with session-specific database"""
        if 'db_id' not in session:
            session['db_id'] = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        db_path = os.path.join(self.database_folder, f"data_{session['db_id']}.db")
        return sqlite3.connect(db_path)

    def validate_sql_query(self, sql: str) -> bool:
        """Basic SQL injection prevention"""
        # Convert to lowercase for checking
        sql_lower = sql.lower().strip()
        
        # Block dangerous SQL keywords
        dangerous_keywords = [
            'drop', 'delete', 'insert', 'update', 'alter', 'create', 'truncate',
            'exec', 'execute', 'sp_', 'xp_', '--', ';--', '/*', '*/'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in sql_lower:
                return False
                
        # Must start with SELECT
        if not sql_lower.startswith('select'):
            return False
            
        return True

    def get_table_schema(self, conn: sqlite3.Connection, table_name: str) -> Dict:
        """Get detailed table schema information"""
        try:
            # Get column information
            schema_df = pd.read_sql(f"PRAGMA table_info({table_name})", conn)
            
            # Get sample data to understand data types better
            sample_df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 3", conn, parse_dates=True)
            
            # Get row count
            count_df = pd.read_sql(f"SELECT COUNT(*) as total_rows FROM {table_name}", conn)
            total_rows = count_df.iloc[0]['total_rows']
            
            columns_info = []
            for _, col in schema_df.iterrows():
                col_name = col['name']
                col_type = col['type']
                
                # Analyze sample data for better type detection
                if col_name in sample_df.columns:
                    sample_values = sample_df[col_name].dropna().head(3).tolist()
                    
                    # Check if it's actually numeric
                    if pd.api.types.is_numeric_dtype(sample_df[col_name]):
                        col_type = 'NUMERIC'
                    elif pd.api.types.is_datetime64_any_dtype(sample_df[col_name]):
                        col_type = 'DATE'
                    
                    columns_info.append({
                        'name': col_name,
                        'type': col_type,
                        'sample_values': sample_values
                    })
            
            return {
                'table_name': table_name,
                'total_rows': total_rows,
                'columns': columns_info
            }
        except Exception as e:
            logger.error(f"Error getting schema: {e}")
            return {}

    def generate_sql_and_explanation(self, query: str, schema_info: Dict) -> Dict:
        """Generate SQL query and explanation using Anthropic Claude"""
        try:
            # Check if Anthropic client is available
            if not self.anthropic_client:
                return {
                    "sql": "",
                    "explanation": "Anthropic Claude client is not available. Please check your API configuration.",
                    "chart_type": "table",
                    "insights": "AI features are currently unavailable."
                }
            
            prompt = f"""You are a data analyst that converts natural language questions into SQL queries.

Database Schema:
Table: {schema_info['table_name']}
Total Rows: {schema_info['total_rows']}
Columns:
{chr(10).join([f"- {col['name']} ({col['type']}): sample values {col['sample_values']}" for col in schema_info['columns']])}

User Question: "{query}"

Instructions:
1. Generate a SQL SELECT query that answers the question
2. Use appropriate aggregations, filters, and sorting
3. Limit results to reasonable numbers (use LIMIT if needed)
4. For time-based data, consider appropriate grouping
5. Use column names exactly as shown in the schema

Return ONLY a JSON object with these keys:
- "sql": The SQL query (without any markdown formatting)
- "explanation": Clear explanation of what the query does
- "chart_type": Suggest the best chart type (bar, line, pie, scatter, table)
- "insights": Brief insight about what this analysis might reveal

Example response:
{{"sql": "SELECT column1, COUNT(*) FROM table_name GROUP BY column1 ORDER BY COUNT(*) DESC LIMIT 10", "explanation": "This query counts occurrences...", "chart_type": "bar", "insights": "This will show which categories are most common..."}}"""
            
            # Make API call to Claude
            response = self.anthropic_client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            result_text = response.content[0].text.strip()
            
            # Clean up the response - remove any markdown formatting
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            
            # Sometimes Claude wraps JSON in extra text, try to extract it
            if '{' in result_text and '}' in result_text:
                start = result_text.find('{')
                end = result_text.rfind('}') + 1
                result_text = result_text[start:end]
            
            result = json.loads(result_text)
            
            # Validate the SQL
            if not self.validate_sql_query(result.get('sql', '')):
                return {
                    "sql": "",
                    "explanation": "Generated query contains potentially unsafe operations.",
                    "chart_type": "table",
                    "insights": "Query validation failed for security reasons."
                }
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Raw response: {result_text}")
            return {
                "sql": "",
                "explanation": f"Error parsing AI response: {str(e)}",
                "chart_type": "table",
                "insights": "Failed to generate analysis."
            }
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return {
                "sql": "",
                "explanation": f"Error generating query: {str(e)}",
                "chart_type": "table",
                "insights": "Failed to generate analysis."
            }

    def create_visualization(self, df: pd.DataFrame, chart_type: str, query: str) -> str:
        """Create appropriate visualization based on data and suggested chart type"""
        try:
            if df.empty:
                fig = go.Figure()
                fig.add_annotation(text="No data to display", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return json.dumps(fig, cls=PlotlyJSONEncoder)
            
            # Clean column names for better display
            df.columns = [col.replace('_', ' ').title() for col in df.columns]
            
            if chart_type == "bar" and len(df.columns) >= 2:
                fig = px.bar(df, x=df.columns[0], y=df.columns[1], 
                           title=f"Bar Chart: {query}")
                
            elif chart_type == "line" and len(df.columns) >= 2:
                fig = px.line(df, x=df.columns[0], y=df.columns[1], 
                            title=f"Line Chart: {query}")
                
            elif chart_type == "pie" and len(df.columns) >= 2:
                fig = px.pie(df, names=df.columns[0], values=df.columns[1], 
                           title=f"Pie Chart: {query}")
                
            elif chart_type == "scatter" and len(df.columns) >= 2:
                color_col = df.columns[2] if len(df.columns) > 2 else None
                fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color=color_col,
                               title=f"Scatter Plot: {query}")
                
            else:
                # Default to table view for complex data
                fig = go.Figure(data=[go.Table(
                    header=dict(values=list(df.columns), fill_color='paleturquoise', align='left'),
                    cells=dict(values=[df[col] for col in df.columns], fill_color='lavender', align='left')
                )])
                fig.update_layout(title=f"Data Table: {query}")
            
            # Update layout for better appearance
            fig.update_layout(
                title_font_size=16,
                height=400,
                margin=dict(l=50, r=50, t=60, b=50)
            )
            
            return json.dumps(fig, cls=PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            # Return simple table as fallback
            fig = go.Figure(data=[go.Table(
                header=dict(values=["Error"], fill_color='red', align='left'),
                cells=dict(values=[[f"Visualization error: {str(e)}"]], fill_color='pink', align='left')
            )])
            return json.dumps(fig, cls=PlotlyJSONEncoder)

    def index(self):
        """Render the main interface"""
        conn = self.get_database_connection()
        table_preview = None
        available_tables = []
        
        try:
            # Check for existing tables
            tables_df = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
            if not tables_df.empty:
                available_tables = tables_df['name'].tolist()
                
                # Show preview of the first table
                table_name = available_tables[0]
                preview_df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 5", conn)
                table_preview = preview_df.to_html(classes='table table-striped table-sm', index=False)
                
        except Exception as e:
            logger.error(f"Error loading table preview: {e}")
        finally:
            conn.close()
        
        return render_template_string(HTML_TEMPLATE, 
                                    table_preview=table_preview, 
                                    available_tables=available_tables)

    def upload_file(self):
        """Handle file upload with better error handling"""
        try:
            if 'file' not in request.files:
                return jsonify({"error": "No file uploaded"}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            if not file.filename.lower().endswith('.csv'):
                return jsonify({"error": "Only CSV files are supported"}), 400
            
            # Secure filename
            filename = secure_filename(file.filename)
            
            # Read and process CSV
            df = pd.read_csv(file.stream, encoding='utf-8')
            
            if df.empty:
                return jsonify({"error": "CSV file is empty"}), 400
            
            # Clean column names for SQL compatibility
            df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col.strip()).lower() for col in df.columns]
            df.columns = [col.strip('_') for col in df.columns]  # Remove leading/trailing underscores
            
            # Handle missing values
            df = df.fillna('')
            
            # Store in database
            conn = self.get_database_connection()
            table_name = f"data_{filename.split('.')[0]}"
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
            
            return jsonify({
                "message": f"File uploaded successfully. {len(df)} rows loaded.",
                "table_name": table_name,
                "columns": list(df.columns),
                "row_count": len(df)
            }), 200
            
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500

    def handle_query(self):
        """Process natural language query with enhanced error handling"""
        try:
            data = request.get_json()
            query = data.get('query', '').strip()
            
            if not query:
                return jsonify({"error": "No query provided"}), 400
            
            conn = self.get_database_connection()
            
            # Get available tables
            tables_df = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
            if tables_df.empty:
                return jsonify({"error": "No data available - please upload a dataset first"}), 400
            
            # Use the first available table (in a full app, you might let users choose)
            table_name = tables_df.iloc[0]['name']
            
            # Get schema information
            schema_info = self.get_table_schema(conn, table_name)
            
            if not schema_info:
                return jsonify({"error": "Could not analyze table schema"}), 400
            
            # Generate SQL and explanation
            ai_response = self.generate_sql_and_explanation(query, schema_info)
            sql_query = ai_response.get('sql', '')
            
            if not sql_query:
                return jsonify({
                    "error": "Could not generate SQL query",
                    "explanation": ai_response.get('explanation', 'Unknown error'),
                    "suggestions": ["Try rephrasing your question", "Be more specific about what you want to analyze"]
                }), 400
            
            # Execute SQL query
            try:
                result_df = pd.read_sql(sql_query, conn)
                
                if result_df.empty:
                    return jsonify({
                        "insight": "No data matches your query criteria.",
                        "explanation": ai_response.get('explanation', ''),
                        "sql": sql_query,
                        "chart_json": self.create_visualization(result_df, "table", query),
                        "data_preview": [],
                        "row_count": 0
                    })
                
                # Generate visualization
                chart_type = ai_response.get('chart_type', 'table')
                chart_json = self.create_visualization(result_df, chart_type, query)
                
                # Create data preview (limit to first 10 rows for display)
                preview_df = result_df.head(10)
                data_preview = preview_df.to_dict(orient='records')
                
                # Generate insight
                insights = ai_response.get('insights', 'Analysis completed successfully.')
                row_count = len(result_df)
                
                if row_count == 1:
                    insight = f"Found 1 result: {insights}"
                else:
                    insight = f"Found {row_count} results: {insights}"
                
                return jsonify({
                    "insight": insight,
                    "explanation": ai_response.get('explanation', ''),
                    "sql": sql_query,
                    "chart_json": chart_json,
                    "data_preview": data_preview,
                    "row_count": row_count,
                    "chart_type": chart_type
                })
                
            except Exception as e:
                logger.error(f"SQL execution error: {e}")
                return jsonify({
                    "error": "Error executing SQL query",
                    "details": str(e),
                    "sql": sql_query,
                    "suggestions": ["Check if column names in your question match the data", "Try a simpler query"]
                }), 400
                
        except Exception as e:
            logger.error(f"Query handling error: {e}")
            return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
        finally:
            try:
                conn.close()
            except:
                pass

    def get_tables(self):
        """Get list of available tables"""
        try:
            conn = self.get_database_connection()
            tables_df = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
            conn.close()
            
            tables = []
            for _, row in tables_df.iterrows():
                table_name = row['name']
                conn = self.get_database_connection()
                schema_info = self.get_table_schema(conn, table_name)
                conn.close()
                tables.append(schema_info)
            
            return jsonify({"tables": tables})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def reset_data(self):
        """Reset/clear all data"""
        try:
            if 'db_id' in session:
                db_path = os.path.join(self.database_folder, f"data_{session['db_id']}.db")
                if os.path.exists(db_path):
                    os.remove(db_path)
                session.pop('db_id', None)
            
            return jsonify({"message": "Data cleared successfully"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def run(self, debug=None, host=None, port=None):
        """Run the Flask application with environment variable support"""
        # Use environment variables with fallbacks
        debug = debug if debug is not None else os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
        host = host or os.environ.get('HOST', '0.0.0.0')
        port = int(port or os.environ.get('PORT', 5000))
        
        logger.info(f"Starting DataGPT Analytics on {host}:{port}")
        logger.info(f"Debug mode: {debug}")
        
        self.app.run(debug=debug, host=host, port=port)

# Modern, responsive HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataGPT Analytics</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #4f46e5;
            --secondary-color: #f8fafc;
            --accent-color: #10b981;
            --text-dark: #1f2937;
            --border-color: #e5e7eb;
        }
        
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .main-container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
            margin: 2rem auto;
            max-width: 1200px;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, var(--primary-color), #6366f1);
            color: white;
            padding: 2rem;
            text-align: center;
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        .header p {
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }
        
        .content-area {
            padding: 2rem;
        }
        
        .upload-section, .query-section {
            background: var(--secondary-color);
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border-color);
        }
        
        .section-title {
            color: var(--text-dark);
            margin-bottom: 1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .btn-custom {
            background: var(--primary-color);
            border: none;
            border-radius: 8px;
            color: white;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .btn-custom:hover {
            background: #4338ca;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        }
        
        .form-control {
            border-radius: 8px;
            border: 1px solid var(--border-color);
            padding: 0.75rem;
        }
        
        .form-control:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }
        
        .insight-box {
            background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
            border: 1px solid #0ea5e9;
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 1.5rem;
        }
        
        .sql-box {
            background: #1f2937;
            color: #f9fafb;
            border-radius: 8px;
            padding: 1rem;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            overflow-x: auto;
        }
        
        .chart-container {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-top: 1rem;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 2rem;
        }
        
        .spinner {
            border: 4px solid #f3f4f6;
            border-top: 4px solid var(--primary-color);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .data-preview {
            max-height: 300px;
            overflow-y: auto;
            border-radius: 8px;
        }
        
        .alert-custom {
            border-radius: 8px;
            border: none;
            padding: 1rem;
        }
        
        .badge-custom {
            background: var(--accent-color);
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 6px;
            font-size: 0.8rem;
        }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="header">
            <h1><i class="fas fa-brain"></i> DataGPT Analytics</h1>
            <p>Ask questions about your data in natural language - Powered by JADA</p>
        </div>
        
        <div class="content-area">
            <!-- Upload Section -->
            <div class="upload-section">
                <h3 class="section-title">
                    <i class="fas fa-upload"></i> Upload Your Dataset
                </h3>
                
                <form id="uploadForm" enctype="multipart/form-data">
                    <div class="row align-items-end">
                        <div class="col-md-8">
                            <input type="file" id="fileInput" class="form-control" accept=".csv" required>
                        </div>
                        <div class="col-md-4">
                            <button type="submit" class="btn btn-custom w-100">
                                <i class="fas fa-cloud-upload-alt"></i> Upload
                            </button>
                        </div>
                    </div>
                </form>
                
                <div id="uploadStatus" class="mt-3"></div>
                
                {% if table_preview %}
                <div class="mt-4">
                    <h5><i class="fas fa-table"></i> Data Preview</h5>
                    <div class="data-preview">
                        {{ table_preview|safe }}
                    </div>
                    {% if available_tables %}
                    <div class="mt-2">
                        <span class="badge-custom">{{ available_tables|length }} table(s) loaded</span>
                    </div>
                    {% endif %}
                </div>
                {% endif %}
            </div>
            
            <!-- Query Section -->
            <div class="query-section">
                <h3 class="section-title">
                    <i class="fas fa-comments"></i> Ask About Your Data
                </h3>
                
                <form id="queryForm">
                    <div class="row">
                        <div class="col-12">
                            <textarea id="queryInput" class="form-control" rows="3" 
                                    placeholder="Examples:&#10;• What are the top 5 products by sales?&#10;• Show me monthly trends in revenue&#10;• Which customers have the highest orders?&#10;• What's the average price by category?"></textarea>
                        </div>
                    </div>
                    <div class="row mt-3">
                        <div class="col-md-8">
                            <button type="submit" class="btn btn-custom">
                                <i class="fas fa-search"></i> Analyze
                            </button>
                            <button type="button" id="resetBtn" class="btn btn-outline-secondary ms-2">
                                <i class="fas fa-trash"></i> Clear Data
                            </button>
                        </div>
                    </div>
                </form>
                
                <div id="loading" class="loading">
                    <div class="spinner"></div>
                    <p>Analyzing your data...</p>
                </div>
                
                <div id="results"></div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // File upload handler
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            const statusDiv = document.getElementById('uploadStatus');
            
            if (!file) {
                showStatus('Please select a file', 'danger');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            showStatus('Uploading and processing file...', 'info');
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    showStatus(`✅ ${result.message}`, 'success');
                    setTimeout(() => location.reload(), 1500);
                } else {
                    showStatus(`❌ ${result.error}`, 'danger');
                }
            } catch (error) {
                showStatus(`❌ Upload failed: ${error.message}`, 'danger');
            }
        });
        
        // Query handler
        document.getElementById('queryForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const query = document.getElementById('queryInput').value.trim();
            if (!query) {
                alert('Please enter a question about your data');
                return;
            }
            
            showLoading(true);
            document.getElementById('results').innerHTML = '';
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query })
                });
                
                const result = await response.json();
                showLoading(false);
                
                if (response.ok) {
                    displayResults(result, query);
                } else {
                    displayError(result);
                }
            } catch (error) {
                showLoading(false);
                displayError({ error: `Request failed: ${error.message}` });
            }
        });
        
        // Reset data handler
        document.getElementById('resetBtn').addEventListener('click', async () => {
            if (confirm('Are you sure you want to clear all data?')) {
                try {
                    const response = await fetch('/reset', { method: 'POST' });
                    const result = await response.json();
                    
                    if (response.ok) {
                        showStatus('✅ Data cleared successfully', 'success');
                        setTimeout(() => location.reload(), 1000);
                    } else {
                        showStatus(`❌ ${result.error}`, 'danger');
                    }
                } catch (error) {
                    showStatus(`❌ Clear failed: ${error.message}`, 'danger');
                }
            }
        });
        
        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
        }
        
        function showStatus(message, type) {
            const statusDiv = document.getElementById('uploadStatus');
            statusDiv.innerHTML = `<div class="alert alert-${type} alert-custom">${message}</div>`;
        }
        
        function displayResults(result, query) {
            const resultsDiv = document.getElementById('results');
            
            let html = `
                <div class="insight-box">
                    <h4><i class="fas fa-lightbulb"></i> Insight</h4>
                    <p class="mb-3">${result.insight}</p>
                    
                    <h5><i class="fas fa-info-circle"></i> How this was calculated:</h5>
                    <p class="mb-3">${result.explanation}</p>
                    
                    <h5><i class="fas fa-code"></i> Generated SQL:</h5>
                    <div class="sql-box">${result.sql}</div>
                    
                    ${result.row_count ? `<div class="mt-3"><span class="badge-custom">${result.row_count} rows found</span></div>` : ''}
                </div>
                
                <div class="chart-container">
                    <div id="chart"></div>
                </div>
            `;
            
            resultsDiv.innerHTML = html;
            
            // Render the chart
            try {
                const chartData = JSON.parse(result.chart_json);
                Plotly.newPlot('chart', chartData.data, chartData.layout, {responsive: true});
            } catch (error) {
                document.getElementById('chart').innerHTML = `<div class="alert alert-warning">Chart rendering error: ${error.message}</div>`;
            }
        }
        
        function displayError(result) {
            const resultsDiv = document.getElementById('results');
            
            let html = `
                <div class="alert alert-danger alert-custom">
                    <h5><i class="fas fa-exclamation-triangle"></i> Error</h5>
                    <p>${result.error}</p>
                    ${result.details ? `<small>Details: ${result.details}</small>` : ''}
                    ${result.sql ? `<div class="mt-2"><strong>Generated SQL:</strong><br><code>${result.sql}</code></div>` : ''}
                </div>
            `;
            
            if (result.suggestions) {
                html += `
                    <div class="alert alert-info alert-custom">
                        <h6>Suggestions:</h6>
                        <ul class="mb-0">
                            ${result.suggestions.map(s => `<li>${s}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }
            
            resultsDiv.innerHTML = html;
        }
    </script>
</body>
</html>
"""

# Application factory
def create_app():
    """Create and configure the Flask application"""
    return DataAnalyticsApp().app

if __name__ == "__main__":
    # Check if .env file exists
    if not os.path.exists('.env'):
        logger.warning("No .env file found. Please create one using the provided .env template.")
        logger.warning("The application may not work correctly without proper environment variables.")
    
    app = create_app()
    app.run()