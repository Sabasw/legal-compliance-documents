import smtplib
from smtplib import SMTP
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

SMTP_SERVER = "smtp.gmail.com" 
SMTP_PORT = 587
SENDER_EMAIL = "sabasaeed410@gmail.com"
SENDER_PASSWORD = "xwfw lkbg uwws ulnt"

def send_thank_you_email(recipient_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        # server.starttls()
        with smtplib.SMTP_SSL("smtp.ionos.de", 465)  as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
            server.quit()

        print("Thank-you email sent successfully!")

    except Exception as e:
        print(f"Failed to send thank-you email: {e}")

def send_reset_password_email(recipient_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Use SMTP_SSL instead of SMTP for port 465
        # server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)  
        # server.login(SENDER_EMAIL, SENDER_PASSWORD)
        # server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        # server.quit()
        # with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        with smtplib.SMTP_SSL("smtp.ionos.de", 465)  as server:
            # server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        print("Reset password email sent successfully!")

    except Exception as e:
        print(f"Failed to send reset password email: {e}")        
        
def send_registration_otp(recipient_email, subject, otp):
    """
    Sends a registration OTP email to the recipient.

    Args:
        recipient_email (str): The recipient's email address.
        subject (str): The subject of the email.
        otp (str): The OTP to include in the email body.

    Returns:
        None
    """
    try:
        # Create the email body with OTP
        # body = f"Dear User,\n\nYour OTP for registration is: {otp}\n\nThis OTP is valid for 5 minutes.\n\nThank you!"
        body = """
Dear User,

Your One-Time Password (OTP) for registration is: {otp}

Please note that this OTP is only valid for 5 minutes.

Thank you for choosing Solasolution!

Kind regards

        """
        # Prepare email message
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        
        # Connect to the SMTP server
        # with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        with smtplib.SMTP_SSL("smtp.ionos.de", 465)  as server:
            # server.starttls()  # Start TLS for secure connection
            server.login(SENDER_EMAIL, SENDER_PASSWORD)  # Login to the email server
            # server.send_message(msg)  # Send the email
            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
            server.quit()
        
        print("Registration email sent successfully!")

    except smtplib.SMTPException as smtp_err:
        print(f"SMTP error occurred: {smtp_err}")
    except Exception as e:
        print(f"Failed to send registration email: {e}")
