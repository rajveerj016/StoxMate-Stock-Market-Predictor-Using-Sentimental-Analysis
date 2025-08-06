from flask import Flask, request, jsonify, render_template, session, redirect
from pymongo import MongoClient
import firebase_admin
from firebase_admin import credentials, auth
from firebase_admin.exceptions import FirebaseError
import smtplib
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


app = Flask(__name__)
app.secret_key = 'my_complex_secret_key_12345'  


client = MongoClient('localhost', 27017)
db = client['Stock_Predictor_Users']
collection = db['Info']


cred = credentials.Certificate('C:\\Users\\Satvik Pandey\\Downloads\\Stock Market Predictor\\templates\\stock-predictor-3fc23-firebase-adminsdk-rb69n-66f71dca22.json')
firebase_admin.initialize_app(cred)


EMAIL_ADDRESS = 'satvikofficial20@gmail.com'  
EMAIL_PASSWORD = 'dtkxcxxgfvibhquj'      
SMTP_SERVER = 'smtp.gmail.com'              
SMTP_PORT = 587                             

def send_otp(email, otp):
    subject = 'Your OTP Code'
    body = f'Your OTP is: {otp}'


    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
       
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)  
            server.send_message(msg)  
        print("OTP sent successfully!")
    except Exception as e:
        print(f"Failed to send OTP: {e}")

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/submit_signup', methods=['POST'])
def submit_signup():
   
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    contact = data.get('contact')
    pan = data.get('pan')

    
    try:
        collection.insert_one({
            'name': name,
            'email': email,
            'contact': contact,
            'pan': pan
        })
        print("User  data inserted into MongoDB successfully.")
    except Exception as e:
        return jsonify(success=False, message=f'Error saving user data to MongoDB: {str(e)}')

  
    otp = random.randint(100000, 999999)
    session['otp'] = otp  
    session['email'] = email  

    
    send_otp(email, otp)

    return jsonify(success=True, message='OTP sent to your email!')

@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    user_otp = request.json.get('otp')
    if user_otp == str(session.get('otp')):
        email = session.get('email')

        
        try:
            user = auth.create_user(
                email=email,
                password="Satvik_123"  
            )
            print(f'Successfully created Firebase user: {user.uid}')
        except FirebaseError as e:
            return jsonify(success=False, message=f'Error creating Firebase user: {str(e)}')

        
        return jsonify(success=True, message='User  registered successfully!', redirect_url='http://localhost:8501')
    else:
        return jsonify(success=False, message='Invalid OTP!')

@app.route('/success')
def success():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)