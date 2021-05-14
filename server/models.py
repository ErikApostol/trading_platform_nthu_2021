from flask import request
from flask_login import UserMixin
from datetime import datetime
from __init__ import db


class Visit(db.Model):
    __tablename__ = 'Visit'
    id = db.Column(db.Integer, primary_key=True)
    ip = db.Column(db.String(1000), unique=True)
    insert_time = db.Column(db.DateTime, default=datetime.now)

    def __init__(self, ip):
        self.ip = ip

class User(UserMixin, db.Model):
    __tablename__ = 'User'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(1000))
    personal_id = db.Column(db.String(1000))
    b_date = db.Column(db.DateTime)
    insert_time = db.Column(db.DateTime, default=datetime.now)


class ForumComment(db.Model):
    __tablename__ = 'ForumComment'
    comment_id = db.Column(db.Integer, primary_key=True)
    comment = db.Column(db.String(2048))
    insert_time = db.Column(db.DateTime, default=datetime.now)
    user_id = db.Column(db.Integer)
    post_id = db.Column(db.Integer)
    user_name = db.Column(db.String(100))

    def __init__(self, comment, user_id, post_id, user_name):
        self.comment = comment
        self.user_id = user_id
        self.post_id = post_id
        self.user_name = user_name


class ForumPost(db.Model):
    __tablename__ = 'ForumPost'
    post_id = db.Column(db.Integer, primary_key=True)
    comment = db.Column(db.String(2048))
    title = db.Column(db.String(100))
    insert_time = db.Column(db.DateTime, default=datetime.now)
    user_id = db.Column(db.Integer)
    video_id = db.Column(db.Integer)

    # tag: taiwan,car,...
    tag = db.Column(db.String(2048))

    def __init__(self, title, comment, user_id, video_id, tag):
        self.title = title
        self.comment = comment
        self.user_id = user_id
        self.video_id = video_id
        self.tag = tag


class UploadFile(db.Model):
    __tablename__ = 'UploadFile'
    file_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)

    vidoe_filename = db.Column(db.String(2048))
    vidoe_hash_filename = db.Column(db.String(2048))
    g_sensor_filename = db.Column(db.String(2048))
    g_sensor_hash_filename = db.Column(db.String(2048))

    accident_time = db.Column(db.String(2048))
    car_to_motor = db.Column(db.String(2048))
    ownership = db.Column(db.String(2048))
    object_hit = db.Column(db.String(2048))
    country = db.Column(db.String(2048))
    description = db.Column(db.String(2048))
    crush_type = db.Column(db.String(2048))
    role = db.Column(db.String(2048))

    insert_time = db.Column(db.DateTime, default=datetime.now)
    analysis_state = db.Column(db.String(20), default='WAITING')
    analysis_result = db.Column(db.String(100), default='')

    def __init__(self, user_id, filename, hash_filename, g_sensor_filename, hash_g_sensor_filename, accident_time,
                 car_or_motor, ownership, object_hit, country, description, crush_type, role):
        self.user_id = user_id
        self.vidoe_filename = filename
        self.vidoe_hash_filename = hash_filename
        self.g_sensor_filename = g_sensor_filename
        self.g_sensor_hash_filename = hash_g_sensor_filename
        self.accident_time = accident_time
        self.car_to_motor = car_or_motor
        self.ownership = ownership
        self.object_hit = object_hit
        self.country = country
        self.description = description
        self.crush_type = crush_type
        self.role = role
