from flask import Flask, jsonify, request
import psycopg2

app = Flask(__name__)

# Database connection details
DB_NAME = "IITGN"
DB_USER = "postgres"
DB_PASSWORD = "Suresh@11"
DB_HOST = "localhost"
DB_PORT = "5432"

# Function to connect to PostgreSQL
def connect_db():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

# Route to display all tables as clickable links
@app.route('/')
def list_tables():
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        tables = [table[0] for table in cursor.fetchall()]
        conn.close()

        # Create HTML with clickable links
        html = "<h2>Available Tables</h2><ul>"
        for table in tables:
            html += f"<li><a href='/table/{table}' target='_blank'>{table}</a></li>"
        html += "</ul>"
        return html
    except Exception as e:
        return f"<p>Error: {str(e)}</p>"

# Route to fetch table data in JSON format
@app.route('/table/<table_name>', methods=['GET'])
def get_table_data(table_name):
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name};")
        rows = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        conn.close()

        # Convert data to JSON format
        table_data = {
            "columns": colnames,
            "data": [dict(zip(colnames, row)) for row in rows],
            "table": table_name
        }
        return jsonify(table_data)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)