from flask import render_template, url_for, flash, redirect,request
from flask_login import login_required, current_user,logout_user
from werkzeug.utils import secure_filename
from auth_app import app
import os
from pdf2image import convert_from_path
from auth_app import db, bcrypt
from auth_app.user import user
from auth_app.user.forms import  ChangePasswordForm, UpdateProfilePictureForm,UploadForm,verifiedForm,verifyuser,RemoveHistory
from auth_app.user.utils import save_picture
from auth_app.user.models import User, Message,ImageSegment
import argparse, os,json
from tqdm import tqdm
from PIL import Image, ImageDraw
import torch
import numpy as np
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from auth_app.recognizor.read import text_recognizer
from auth_app.hindi_detection.runonnx import run_inference_hindi
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html')
    


def run_inference(file,name,message):
    
    parser = argparse.ArgumentParser(description="Text Line Detection Inference")
    parser.add_argument( "--config-file", default="inference_config.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument( "opts",help="Modify config options using the command-line",default=None,nargs=argparse.REMAINDER)
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    # logger.info("Using {} GPUs".format(num_gpus))
    
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    
    output_dir = cfg.OUTPUT_DIR
    last_checkpoint_file= cfg.LAST_CHECKPOINT_FILE

    checkpointer = DetectronCheckpointer(cfg, model, last_checkpoint_file=last_checkpoint_file,save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)
    
    test_img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], name))
    test_w,test_h = test_img.size
    json_dict = {"images": [{"height": test_h,"width": test_w,"id": 0,"file_name": os.path.join(app.config['UPLOAD_FOLDER'], name)}],
    "annotations": [],
    "categories": [{"supercategory": "urdu_text","id": 1, "name": "text"}]
    }

    json.dump(json_dict,open("maskrcnn_benchmark/engine/test.json", 'w'), ensure_ascii=False)
    
    iou_types = ("bbox",)
    if cfg.MODEL.BOUNDARY_ON:
        iou_types = iou_types + ("bo",)
    output_folders = [cfg.OUTPUT_DIR]*len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST



    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = "vis"
            if os.path.exists(output_folder):
                print("Output folder already exists. Overwriting...")
                import shutil
                shutil.rmtree(output_folder)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        bo = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()
    
    bboxes = []
    for detection in bo:
        bboxes.append(detection["seg_rorect"])
    bboxes = sorted(bboxes,key=lambda x:x[1])

    image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], name))
    image = image.convert('RGB')
    
    # Draw the bounding boxes on the image
    draw = ImageDraw.Draw(image)
    uploaded_image_width, uploaded_image_height = image.size
    compatible_line_width = int(10//(2000/uploaded_image_width)) # Line width 10 best for Image Height 2000
    texts=''
    for num_box,points in enumerate(tqdm(bboxes)):
        min_x = min(points[0::2])
        max_x = max(points[0::2])
        min_y = min(points[1::2])
        max_y = max(points[1::2])


        # For drawing the bounding boxes
        color = tuple([int(x) for x in np.random.randint(0, 256, 3)])
        draw.rectangle(((min_x, min_y), (max_x, max_y)), outline=color, width=compatible_line_width)

        # For saving the cropped image
        im_cropped = image.crop((int(min_x), int(min_y), int(max_x), int(max_y)))
        im_cropped.save(os.path.join(app.config['UPLOAD_FOLDER'], str(num_box) + "_" + name))
        imagename = 'uploads/' +str(num_box) + "_" + name
        text = text_recognizer(im_cropped)
        texts=texts+text
        imagesegment = ImageSegment(message_id=message.id, segment_image=imagename, text=text, text_modified=text)
        db.session.add(imagesegment)
        db.session.commit()
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], name))
    message.image_name=name
    message.text=texts
    message.text_modified=texts
    db.session.commit()



@user.route('/choosefile', methods=['GET', 'POST'])
@login_required
def choosefile():
    form = UploadForm()
    if request.method == 'POST' and form.validate():
        language = form.select.data
        file=form.file.data

        if file.filename == '':
            flash('No selected file', 'danger')
            return render_template('choosefile.html',form=form)
        
        elif file.filename in [file.image_name for file in Message.query.filter_by(user_id=current_user.id)]:
            flash('This File was previously Loaded.If you want to load again Then Either delete the file from history or rename it('+file.filename+')','warning')
            message=Message.query.filter_by(user_id=current_user.id).filter_by(image_name=file.filename).first()
            imagesegment=ImageSegment.query.filter_by(message_id=message.id).all()
            return render_template('upload.html',title='Upload',message=message,imagesegments=imagesegment)
        
        elif file.filename.endswith('.pdf'):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            path=os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            images = convert_from_path(path)
            
            for index, image in enumerate(images):
                filename=file.filename.split('.')[0]+'-'+str(index)+'.png'
                image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                name=filename
                text_modified="hy"
                message = Message(user_id=current_user.id, image_name=filename,text=text_modified,text_modified=text_modified)
                db.session.add(message)
                db.session.commit()
                if language=='Urdu' or language=='1':
                    run_inference(file,name,message)
                    message.model_used='URDU'
                elif language=='Hindi' or language=='2':
                    run_inference_hindi(file,name,message)
                    message.model_used='HINDI'
                if(index==0):
                    id=message.id
            message=Message.query.filter_by(id=id).first()
            imagesegments=ImageSegment.query.filter_by(message_id=id) 
            return render_template('upload.html',title='Upload',message=message,imagesegments=imagesegments)

        elif file.filename.endswith('.jpg') or file.filename.endswith('.png') or file.filename.endswith('.jpeg'):
            flash('Image Uploaded Successfully', 'success')
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            message = Message(user_id=current_user.id, image_name=file.filename, text=file.filename, text_modified=file.filename)
            db.session.add(message)
            db.session.commit()
            if language=='Urdu' or language=='1':
                run_inference(file,file.filename,message)
                message.model_used='URDU'
                
            elif language=='Hindi' or language=='2':
                run_inference_hindi(file,file.filename,message)
                message.model_used='HINDI'
            db.session.commit()
            imagesegment=ImageSegment.query.filter_by(message_id=message.id).all() 
            return render_template('upload.html',title='Upload',message=message,imagesegments=imagesegment) 
            
        else:
            return render_template('error.html', title='Error')
    return render_template('choosefile.html', title='Choose File', form=form)





@user.route('/upload/<int:id>', methods=['GET', 'POST'])
@login_required
def upload(id):
    if current_user:
        message = Message.query.filter_by(id=id).first_or_404()
        if message.user_id == current_user.id:
            imagesegment = ImageSegment.query.filter_by(message_id=message.id).all()
            return render_template('upload.html', title='Upload', message=message, imagesegments=imagesegment)
        else:
            # If the message does not belong to the current user, return an error response
            return render_template('error.html', title='Error')
    else:
        return render_template('error.html', title='Error')




@user.route('/navigate/<int:id>', methods=['GET', 'POST'])
@login_required
def navigate(id):
    if current_user.is_authenticated and current_user:
        message = Message.query.get(id)
        if message:
            if message.user_id == current_user.id:
                direction = request.form.get('direction', 'next')
                if direction == 'next':
                    next_message = Message.query.filter(Message.id > id, Message.user_id == current_user.id).order_by(Message.id).first()
                    if next_message:
                        return redirect(url_for('user.upload', id=next_message.id))
                elif direction == 'prev':
                    prev_message = Message.query.filter(Message.id < id, Message.user_id == current_user.id).order_by(Message.id.desc()).first()
                    if prev_message:
                        return redirect(url_for('user.upload', id=prev_message.id))
                # If there is no next message, redirect to the same page with the current message id
                return redirect(url_for('user.upload', id=id))
            else:
                # If the message does not belong to the current user, return an error response
                return render_template('error.html', title='Error')
        else:
            # If the requested message does not exist, return an error response
            return render_template('error.html', title='Error')
    else:
        return render_template('error.html', title='Error')


@user.route('/history', methods=['GET', 'POST'])
@login_required
def history():
    form=RemoveHistory()
    user = current_user
    messages = Message.query.filter_by(user_id=user.id).all()
    return render_template('history.html', title='History',form=form, messages=messages,user=user)


@user.route('/make_admin/<string:email>', methods=['GET', 'POST'])
@login_required
def make_admin(email):
    if current_user.is_authenticated:
        if current_user.admin:
            user=User.query.filter_by(email=email).first()
            name=user.username
            if user.admin:
                user.admin=False
                db.session.commit()
                flash(name+' is no longer an admin','success')
                return redirect(url_for('user.admin',email=email))
            else:
                user.admin=True
                db.session.commit()
                flash(name+' is now an admin','success')

                return redirect(url_for('user.admin',email=email))
        else:
            return render_template('error.html', title='Error')
    else:
        return render_template('error.html', title='Error')


@user.route('/delete_history', methods=['GET', 'POST'])
@login_required
def delete_history():
    if request.method == 'POST':
        ids = request.form.getlist('checkbox')
        if not ids:
            flash('No files selected', 'warning')
            return redirect(url_for('user.history'))
        for id in ids:
            message = Message.query.filter_by(id=int(id)).first()
            segments = ImageSegment.query.filter_by(message_id=message.id).all()
            for segment in segments:
                db.session.delete(segment)
                db.session.commit()
            db.session.delete(message)
        db.session.commit()
        flash('Files have been deleted', 'success')
    return redirect(url_for('user.history'))





@user.route('/admin', methods=['GET', 'POST'])
@login_required
def admin():
    if current_user.admin:
        form=verifiedForm()
        users=User.query.all()
        return render_template('admin.html',form=form, title='Admin',users=users)
    else:
        return render_template('error.html', title='Error')




@user.route('/remove/<string:email>', methods=['GET', 'POST'])
@login_required
def remove(email):
    if current_user.admin:
        user = User.query.filter_by(email=email).first()
        name = user.username
        messages = Message.query.filter_by(user_id=user.id).all()
        for message in messages:
            segments = ImageSegment.query.filter_by(message_id=message.id).all()
            for segment in segments:
                db.session.delete(segment)
                db.session.commit()
            db.session.delete(message)
        db.session.delete(user)
        db.session.commit()
        flash(name+' has been removed','success')
        return redirect(url_for('user.admin',email=email))
    else:
        return render_template('error.html', title='Error')

@user.route('/choose_option/<int:id>', methods=['POST'])
@login_required
def choose_option(id):
    if current_user:
        selected_id=request.form.get('segment_selected')
        if not selected_id:
            flash('No segment selected for saving', 'warning')
            return  redirect(url_for('user.upload',id=id))
        if request.form['action'] == 'save':
            new_text = request.form['new_text_' + selected_id]
            if not new_text:
                new_text=""
            imagesegmented=ImageSegment.query.filter_by(id=selected_id).first()
            imagesegmented.text_modified = new_text
            db.session.commit()
            flash('Text has been saved Successfully....."', 'success')
            message=Message.query.filter_by(id=id).first()
            imagesegments=ImageSegment.query.filter_by(message_id=message.id).all()
            texts=""
            for text in imagesegments:
                texts=texts+text.text_modified+'\n'
            message.text_modified=texts
            db.session.commit()
            return redirect(url_for('user.upload',id=id))
        elif request.form['action'] == 'delete':
            imagesegmented=ImageSegment.query.filter_by(id=selected_id).first()
            db.session.delete(imagesegmented)
            db.session.commit()
            flash('Segment has been deleted', 'success')
            return redirect(url_for('user.upload',id=id))
            
        elif request.form['action'] == 'moveup':
            curr_id = int(selected_id)
            prev_id = curr_id - 1
            prev_segment = None
            while prev_id >= 1 and not prev_segment:
                prev_segment = ImageSegment.query.get(prev_id)
                prev_id -= 1
            if prev_segment:
                curr_segment = ImageSegment.query.get(curr_id)
                curr_segment.segment_image, prev_segment.segment_image = prev_segment.segment_image, curr_segment.segment_image
                curr_segment.text, prev_segment.text = prev_segment.text, curr_segment.text
                curr_segment.text_modified, prev_segment.text_modified = prev_segment.text_modified, curr_segment.text_modified
                db.session.commit()
                flash('Segment has been moved up', 'success')
                return redirect(url_for('user.upload', id=id))
            else:
                flash('Segment is already at the top', 'warning')
                return redirect(url_for('user.upload', id=id))

        elif request.form['action'] == 'movedown':
            curr_id = int(selected_id)
            prev_id = curr_id + 1
            prev_segment = None
            while not prev_segment:
                prev_segment = ImageSegment.query.get(prev_id)
                prev_id += 1
                if not prev_segment:
                    # If we reach the end of the list without finding the next segment, break out of the loop
                    if prev_id > ImageSegment.query.count():
                        break
            if prev_segment:
                curr_segment = ImageSegment.query.get(curr_id)
                curr_segment.segment_image, prev_segment.segment_image = prev_segment.segment_image, curr_segment.segment_image
                curr_segment.text, prev_segment.text = prev_segment.text, curr_segment.text
                curr_segment.text_modified, prev_segment.text_modified = prev_segment.text_modified, curr_segment.text_modified
                db.session.commit()
                flash('Segment has been moved down', 'success')
                return redirect(url_for('user.upload', id=id))
            else:
                flash('Segment is already at the bottom', 'warning')
                return redirect(url_for('user.upload', id=id))

    else:
        return render_template('error.html', title='Error')



@user.route('/verify_user/<string:email>', methods=['GET', 'POST'])
@login_required
def verify_user(email):
    if current_user.admin:
        user=User.query.filter_by(email=email).first()
        user.verified=True
        db.session.commit()
        return redirect(url_for('user.verify_users',email=email))
    else:
        return render_template('error.html', title='Error')


@user.route('/verify_users', methods=['GET', 'POST'])
@login_required
def verify_users():
    if current_user.admin:
        form=verifyuser()
        user=User.query.all()
        return render_template('verify_users.html',form=form, title='Verify Users',users=user)
    else:
        return render_template('error.html', title='Error')


@user.route('/restrict_user/<string:email>', methods=['GET', 'POST'])
@login_required
def restrict_user(email):
    if current_user.admin:
        user=User.query.filter_by(email=email).first()
        name=user.username
        if user.enabled:
            user.enabled=False
            db.session.commit()
            flash(name+' has been disabled','success')
            return redirect(url_for('user.admin',email=email))
        elif not user.enabled:
            user.enabled=True
            db.session.commit()
            flash(name+' has been enabled','success')
            return redirect(url_for('user.admin',email=email))
        else:
            return render_template('error.html', title='Error')
    else:
        return render_template('error.html', title='Error')



@user.route('/account', methods=['GET', 'POST'])
@login_required
def account():
    change_password_form = ChangePasswordForm()
    update_profile_picture_form = UpdateProfilePictureForm()

    if change_password_form.validate_on_submit():
        if bcrypt.check_password_hash(current_user.password, change_password_form.old_password.data):
            new_hashed_password = bcrypt.generate_password_hash(change_password_form.new_password.data)
            current_user.password = new_hashed_password
            db.session.commit()
            flash("Password changed successfully", "success")
            return redirect(url_for('auth.logout'))
        else:
            flash("Invalid Password", "danger")
    elif update_profile_picture_form.validate_on_submit():
        profile_picture = save_picture(update_profile_picture_form.picture.data)
        current_user.profile_picture = profile_picture
        db.session.commit()
        flash("Profile Picture Updated!", "success")

    profile_picture = url_for('static', filename='profile_pictures/' + current_user.profile_picture)

    return render_template(
            'account.html',
            title='account',
            profile_picture=profile_picture,
            change_password_form=change_password_form,
            update_profile_picture_form=update_profile_picture_form
        )
