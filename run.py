from flask import Flask, render_template, request, jsonify , redirect , url_for
import os
from werkzeug.utils import secure_filename
from final_image import process_image_query
app = Flask(__name__)

# Create an upload folder if it doesn't exist
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}

def allowed_file(filename, file_type):
    """Check if the uploaded file is allowed based on its type."""
    extension = filename.rsplit(".", 1)[1].lower()
    if file_type == "pdf":
        return extension == "pdf"
    elif file_type == "image":
        return extension in {"png", "jpg", "jpeg"}
    return False

@app.route("/", methods=["GET"])
def home():
    return render_template("base.html")

@app.route("/upload", methods=["GET"])
def upload():
    """Render the upload page with no specific file type."""
    return render_template("upload.html", file_type="pdf")



@app.route('/image_chat',methods=['GET'])
def image_chat():
    return render_template('img_chat.html')





@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    
    """Handles single image upload and passes it to abc() function without saving locally."""
    if request.method == "GET":
        return render_template("upload.html", file_type="image")

    
    print("Hello")
    # Check if a file is provided
    if "files" not in request.files:
        return jsonify({"success": False, "message": "No file provided"}), 400

    files = request.files.getlist("files")  # Get the list of files
    if not files or files[0].filename == "":
        return jsonify({"success": False, "message": "No selected file"}), 400

    # Ensure only one file is uploaded
    if len(files) > 1:
        return jsonify({"success": False, "message": "Only one image is allowed"}), 400

    file = files[0]  # Take the first (and only) file

    # Validate file type
    if not allowed_file(file.filename, "image"):
        return jsonify({"success": False, "message": f"Invalid file type for {file.filename}. Only images (png, jpg, jpeg) are allowed."}), 400
    print('Yes')
    # Call the abc() function with the FileStorage object
    try:
        
        print('o9')
        res1 = process_image_query(file)
        print(res1)
        print('OH')
        return redirect(url_for('image_chat'))
        # Pass the FileStorage object directly to abc()
        # You can use 'result' if abc() returns something useful
    except Exception as e:
        return jsonify({"success": False, "message": f"Error processing image: {str(e)}"}), 500

    # Return success response
    return jsonify({
        "success": True,
        "message": "Image processed successfully!",
        "filename": file.filename
    }), 200




@app.route("/upload-pdf", methods=["GET", "POST"])
def upload_pdf():
    """Handles PDF uploads."""
    if request.method == "GET":
        return render_template("upload.html", file_type="pdf")

    if "files" not in request.files:
        return jsonify({"success": False, "message": "No file provided"}), 400

    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"success": False, "message": "No selected file"}), 400

    # Since the frontend ensures only one PDF is uploaded, take the first file
    file = files[0]

    if not allowed_file(file.filename, "pdf"):
        return jsonify({"success": False, "message": "Invalid file type. Only PDFs are allowed."}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    return jsonify({
        "success": True,
        "message": "PDF uploaded successfully!",
        "filename": filename,
        "path": file_path
    }), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)