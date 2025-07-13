from flask import Flask
from models import db
from migrations.community import upgrade as upgrade_community
from migrations.user_columns import upgrade as upgrade_user_columns
import os

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

def run_migrations():
    with app.app_context():
        # Create instance directory if it doesn't exist
        os.makedirs(os.path.join(basedir, 'instance'), exist_ok=True)
        
        # Run community migrations
        print("Running community migrations...")
        upgrade_community()
        
        # Run user columns migrations
        print("Running user columns migrations...")
        upgrade_user_columns()
        
        print("Migrations completed successfully!")

if __name__ == '__main__':
    run_migrations() 