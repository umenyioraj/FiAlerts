import sys
from pathlib import Path

import resend

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from enviornment.enviornment import RESEND_API_KEY

resend.api_key = RESEND_API_KEY

def test_function(total):
    if total > 10:
        message = resend.Emails.send({
            "from": "onboarding@resend.dev",
            "to": "umenyiorajosh@gmail.com",
            "subject": "Hello World",
            "html": "<p>Congrats on sending your <strong>first email</strong>!</p>"
            })  
        print("Email sent successfully!")
    

test_function(15)

