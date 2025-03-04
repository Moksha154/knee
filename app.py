from flask import Flask, render_template, request,redirect, url_for

from keras.models import load_model
from keras.preprocessing import image
import cv2
app = Flask(__name__)


dic = {0 : 'Normal', 1 : 'Doubtful', 2 : 'Mild', 3 : 'Moderate', 4 : 'Severe'}


#Image Size
img_size=256
model = load_model('model.h5')

model.make_predict_function()

def predict_label(img_path):
    img=cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    resized=cv2.resize(gray,(img_size,img_size)) 
    i = image.img_to_array(resized)/255.0
    i = i.reshape(1,img_size,img_size,1)
    p = model.predict(i)  # Get prediction probabilities
    p = p.argmax(axis=-1)  # Get class with highest probability
    return dic[int(p[0])]
 





# Dummy user credentials
USER_CREDENTIALS = {
    "admin": "password123",
    "user": "test123",
    "Moksha":"Moksha2#"
}

# Variable to track authentication status
is_authenticated = False

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    global is_authenticated  # Use global variable
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            is_authenticated = True
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")

@app.route("/index")
def index():
    if not is_authenticated:
        return redirect(url_for("login"))  # Redirect if not logged in
    return render_template("index.html")

@app.route("/logout")
def logout():
    global is_authenticated
    is_authenticated = False  # Reset authentication status
    return redirect(url_for("home"))

@app.route("/predict", methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "Error: No file uploaded", 400  # Return an error if no file is found

    img = request.files['file']
    if img.filename == '':
        return "Error: No selected file", 400  # Return an error if no file is selected

    img_path = "uploads/" + img.filename    
    img.save(img_path)

    try:
        p = predict_label(img_path)
        print(p)
        return str(p).upper()
    except Exception as e:
        print("Prediction Error:", str(e))
        return "Error: Unable to process the image", 500  # Return an error for processing issues



if __name__ =='__main__':
    #app.debug = True
    app.run(debug = True)
    