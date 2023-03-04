from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, SubmitField,SelectField,validators
from wtforms.validators import DataRequired,  EqualTo, Length





class ChangePasswordForm(FlaskForm):
    old_password = PasswordField('Old Password', validators=[
            DataRequired(),
            Length(min=8, max=120)
        ])

    new_password = PasswordField('New Password', validators=[
            DataRequired(),
            Length(min=8, max=120)
        ])

    new_confirm_password = PasswordField('Confirm New Password', validators=[
            DataRequired(),
            EqualTo('new_password'),
            Length(min=8, max=120)
        ])

    submit = SubmitField('Change Password')


class UpdateProfilePictureForm(FlaskForm):
    picture = FileField('', validators=[
            FileAllowed(['png', 'jpg', 'jpeg'])
        ])

    submit = SubmitField('Update')

class UploadForm(FlaskForm):
    file = FileField('Choose File To Upload', validators=[
            FileAllowed(['pdf', 'png', 'jpg', 'jpeg'])],
            render_kw={"class": "form-control", "required": True})
    
    select = SelectField('Choose language', choices=[('', 'Choose language'), ('1', 'Urdu'), ('2', 'Hindi')],
                         validators=[validators.InputRequired()],
                         render_kw={"class": "form-select", "required": True})

    submit = SubmitField('Upload')

class verifiedForm(FlaskForm):
    submit = SubmitField('Remove')

class verifyuser(FlaskForm):
    # make verify column of user true if button clicked
    submit=SubmitField('Verify')
    remove=SubmitField('Remove')

class RemoveHistory(FlaskForm):
    submit=SubmitField('Remove')

