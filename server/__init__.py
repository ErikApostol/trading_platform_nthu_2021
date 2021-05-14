from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()


def create_app():
    app = Flask(__name__)

    app.config['SECRET_KEY'] = 'nthu_insure_tech_server'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
    app.config['UPLOAD_ROOT'] = '/tmp'

    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    from models import User

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    from auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)

    from main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    from forum import forum as forum_blueprint
    app.register_blueprint(forum_blueprint)

    from file import file as file_blueprint
    app.register_blueprint(file_blueprint)


    return app


if __name__ == "__main__":
    app = create_app()
    # db.create_all(app)
    app.debug = True
    app.run(host='0.0.0.0', port=80)
