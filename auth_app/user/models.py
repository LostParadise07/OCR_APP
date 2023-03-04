from itsdangerous import URLSafeTimedSerializer as Serializer
from flask_login import UserMixin
from auth_app import db, app, login_manager
from datetime import datetime, timedelta

@login_manager.user_loader
def user_loader(user_id):
    return User.query.get(user_id)

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(30), nullable=False, unique=True)
    email = db.Column(db.String(120), nullable=False, unique=True)
    password = db.Column(db.String(60), nullable=False)
    admin = db.Column(db.Boolean, default=False)
    verified = db.Column(db.Boolean, default=False)
    enabled = db.Column(db.Boolean, default=True)
    profile_picture = db.Column(db.String(20), default='default.png')
    messages = db.relationship('Message', backref='user', lazy=True, foreign_keys='Message.user_id')


    def __repr__(self):
        return f"User({self.username}, {self.email})"

    def get_reset_token(self, expiry_period=86400):
        expires_at = int((datetime.utcnow() + timedelta(seconds=expiry_period)).timestamp())
        s = Serializer(app.config['SECRET_KEY'])
        token = s.dumps({'user_id': self.id, 'expires_at': expires_at})
        return token

    @staticmethod
    def verify_reset_token(token):
        s = Serializer(app.config['SECRET_KEY'])
        try:
            data = s.loads(token)
        except:
            return None
        user_id = data.get('user_id')
        expires_at = data.get('expires_at')
        if expires_at < int(datetime.utcnow().timestamp()):
            return None
        return User.query.get(user_id)


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_name = db.Column(db.String(20), nullable=False)
    text = db.Column(db.Text)
    text_modified = db.Column(db.Text)
    model_used=db.Column(db.String(20),default='none')
    tasks=db.Column(db.Integer,default=0)
    tasks_done=db.Column(db.Integer,default=0)

def __repr__(self):
        return f"Message({self.user_id}, {self.image_name})"

class ImageSegment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message_id = db.Column(db.Integer, db.ForeignKey('message.id'), nullable=False)
    segment_image = db.Column(db.String, nullable=False)
    text = db.Column(db.Text)
    text_modified = db.Column(db.Text)

def __repr__(self):
        return f"ImageSegment({self.message_id}, {self.segment_number})"




