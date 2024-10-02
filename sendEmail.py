import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Obtendo as variáveis de ambiente
email_sender = os.getenv("EMAIL_SENDER")
email_recipient = os.getenv("EMAIL_RECIPIENT")
email_password = os.getenv("EMAIL_PASSWORD")

def send_email():

    email_sender = 'bcnascimento2003@gmail.com'
    email_receiver = 'elbrienonascimento2003@gmail.com'
    subject = 'Notificação de Execução do Pipeline'
    body = 'Pipeline executado!'

    msg = MIMEMultipart()
    msg['From'] = email_sender
    msg['To'] = email_receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.example.com', 587) as server:
            server.starttls()
            server.login(email_sender, 'sua_senha')
            server.send_message(msg)
        print("E-mail enviado com sucesso!")
    except Exception as e:
        print(f"Erro ao enviar e-mail: {e}")

if __name__ == "__main__":
    send_email()
