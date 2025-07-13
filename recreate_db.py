import os
from app import db, app
from models.user import User
from models.roadmap import CareerPath, Milestone, Skill, Resource, RoadmapProgress
from models.blog import Post, Comment, Like

def recreate_database():
    """Recreate the database with the current schema."""
    # Check if the database file exists
    db_path = os.path.join(app.instance_path, 'app.db')
    if os.path.exists(db_path):
        print(f"Removing existing database at {db_path}")
        os.remove(db_path)
    
    # Create all tables
    with app.app_context():
        print("Creating database tables...")
        db.create_all()
        print("Database tables created successfully!")

if __name__ == "__main__":
    recreate_database() 