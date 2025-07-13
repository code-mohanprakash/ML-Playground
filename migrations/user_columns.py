from models import db
from sqlalchemy import text

def upgrade():
    """Add github_id and linkedin_id columns to the user table"""
    try:
        # Check if columns already exist
        db.session.execute(text("PRAGMA table_info(user)"))
        columns = db.session.execute(text("PRAGMA table_info(user)")).fetchall()
        column_names = [column[1] for column in columns]
        
        # For SQLite, we need to recreate the table to add UNIQUE constraints
        # First, add the columns without UNIQUE constraint
        if 'github_id' not in column_names:
            db.session.execute(text("ALTER TABLE user ADD COLUMN github_id VARCHAR(100)"))
            print("Added github_id column to user table")
        
        if 'linkedin_id' not in column_names:
            db.session.execute(text("ALTER TABLE user ADD COLUMN linkedin_id VARCHAR(100)"))
            print("Added linkedin_id column to user table")
        
        db.session.commit()
        print("User table migration completed successfully")
    except Exception as e:
        db.session.rollback()
        print(f"Error during user table migration: {str(e)}")
        raise

def downgrade():
    """This is a SQLite database, which doesn't support dropping columns easily.
    To downgrade, we would need to:
    1. Create a new table without these columns
    2. Copy data from the old table
    3. Drop the old table
    4. Rename the new table
    
    This is not implemented for simplicity.
    """
    print("Downgrade not implemented for SQLite. To downgrade, recreate the database.") 