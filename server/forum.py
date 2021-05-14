from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_required, current_user
from sqlalchemy import or_, text

from models import User, ForumComment, ForumPost, UploadFile
from main import home
from __init__ import db

forum = Blueprint('forum', __name__)


@forum.route('/forum/filter', methods=['GET'])
# @login_required
def forum_filter():
    filter_text = request.values.get('filter')

    data = ForumPost.query.order_by(ForumPost.insert_time.desc()).all()

    post_list = []
    for d in data:
        sql = text("SELECT * FROM 'UploadFile' \
        WHERE file_id={video_id} AND ( \
        car_to_motor LIKE '%{filter}%' \
        OR ownership LIKE '%{filter}%' \
        OR object_hit LIKE '%{filter}%' \
        OR country LIKE '%{filter}%' \
        OR crush_type LIKE '%{filter}%' \
        OR role LIKE '%{filter}%' \
        ) \
        ".format(video_id=d.video_id, filter=filter_text))
        result = db.engine.execute(sql).fetchall()
        if len(result) > 0:
            post_list.append(d)

    content_list = []
    for d in post_list:
        content_list.append({
            "id": d.post_id,
            "time": d.insert_time,
            "user_id": d.user_id,
            "user_email": User.query.filter_by(id=d.user_id).first().email,
            "user_name": User.query.filter_by(id=d.user_id).first().name,
            "comment": d.comment,
            "title": d.title,
            "video_id": d.video_id
        })

    return_data = {
        "count": len(post_list),
        "content": content_list
    }

    return render_template('search.html', forum_data=return_data)


@forum.route('/forum')
def forum_index():
    data = ForumPost.query.order_by(ForumPost.insert_time.desc()).all()
    #data = fake_data

    content_list = []
    for d in data:
        content_list.append({
            "id": d.post_id,
            "time": d.insert_time,
            "user_id": d.user_id,
            "user_email": User.query.filter_by(id=d.user_id).first().email,
            "user_name": User.query.filter_by(id=d.user_id).first().name,
            "comment": d.comment,
            "title": d.title,
            "video_id": d.video_id
        })

    return_data = {
        "count": len(data),
        "content": content_list
    }

    return render_template('forum.html', forum_data=return_data)


@forum.route('/users_own_video')
@login_required
def users_own_video():
    data = ForumComment.query.filter_by(user_id=session.get('user_id')).all()

    content_list = []
    for d in data:
        content_list.append({
            "id": d.comment_id,
            "time": d.insert_time,
            "user_id": d.user_id,
            "user_email": User.query.filter_by(id=d.user_id).first().email,
            "user_name": User.query.filter_by(id=d.user_id).first().name,
            "comment": d.comment,
            "title": d.title,
            "video_id": d.video_id
        })

    return_data = {
        "count": len(data),
        "content": content_list
    }

    stats = home()

    return render_template('forum.html', forum_data=return_data, stats=stats)


@forum.route('/get_forum_data')
# @login_required
def get_forum_data():
    data = ForumComment.query.order_by(ForumComment.insert_time.desc()).all()

    content_list = []
    for d in data:
        content_list.append({
            "id": d.comment_id,
            "time": d.insert_time,
            "user_id": d.user_id,
            "title": d.title,
            "user_email": User.query.filter_by(id=d.user_id).first().email,
            "user_name": User.query.filter_by(id=d.user_id).first().name,
            "comment": d.comment,
            "video_id": d.video_id
        })

    return_data = {
        "count": len(data),
        "content": content_list
    }

    return jsonify(return_data)


@forum.route('/post', methods=['POST'])
@login_required
def post_data():
    user_id = session['user_id']
    data = request.get_json()
    comment = data.get('comment')
    video_id = data.get('video_id')
    tag = data.get('tag')
    title = data.get('title')

    new_comment = ForumPost(title=title, comment=comment, user_id=user_id,
                                 video_id=video_id, tag=tag)
    db.session.add(new_comment)
    db.session.commit()

    return redirect(url_for('forum.forum_index'))


@forum.route('/comment', methods=['GET'])
@login_required
def post_comment_data():
    user_id = session['user_id']
    comment = request.values.get('comment')
    post_id = request.values.get('post_id')
    user_name = User.query.filter_by(id=user_id).first().name

    new_comment = ForumComment(comment=comment, user_id=user_id,
                                 post_id=post_id, user_name=user_name)
    db.session.add(new_comment)
    db.session.commit()

    return redirect(url_for('forum.forum_index'))


@forum.route('/post_page', methods=['GET'])
# @login_required
def post_page():
    post_id = int(request.values.get('post_id'))

    post = ForumPost.query.filter_by(post_id=post_id).first()
    comments = ForumComment.query.filter_by(post_id=post_id).order_by(ForumComment.insert_time.asc()).all()
    d = UploadFile.query.filter_by(file_id=post.video_id).first()

    video_content_list = {
        "video_id": d.file_id,
        "user_id": d.user_id,
        "vidoe_filename": d.vidoe_filename,
        "vidoe_hash_filename": d.vidoe_hash_filename,
        "g_sensor_hash_filename": d.g_sensor_hash_filename,
        "accident_time": d.accident_time,
        "car_to_motor": d.car_to_motor,
        "ownership": d.ownership,
        "object_hit": d.object_hit,
        "country": d.country,
        "description": d.description,
        "crush_type": d.crush_type,
        "role": d.role,
        "insert_time": d.insert_time,
        "analysis_state": d.analysis_state,
        "analysis_result": d.analysis_result,
        "user_email": User.query.filter_by(id=d.user_id).first().email,
        "user_name": User.query.filter_by(id=d.user_id).first().name
    }

    post_content = {
        "post_id": post.post_id,
        "time": post.insert_time,
        "user_id": post.user_id,
        "user_email": User.query.filter_by(id=post.user_id).first().email,
        "user_name": User.query.filter_by(id=post.user_id).first().name,
        "comment": post.comment,
        "title": post.title,
        "video_id": post.video_id
    }

    comment_list = []
    for m in comments:
        comment_list.append({
            "comment": m.comment,
            "user_name": m.user_name
        })

    return_data = {
        "video_content": video_content_list,
        "post_content": post_content,
        "comment_content": comment_list,
        "comment_count": len(comment_list)
    }

    # print(return_data)
    return render_template('post_page.html', data=return_data)
