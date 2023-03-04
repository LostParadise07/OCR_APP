from auth_app import run
from auth_app import db
from auth_app.user.models import User
if __name__ == "__main__":
    db.create_all()
    run()
