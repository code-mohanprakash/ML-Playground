from models import db, User, Blog, Comment, Like, Follow, Hashtag
from sqlalchemy import text

def upgrade():
    # Create tables
    db.create_all()
    db.session.commit()

    # Create indexes for better performance
    try:
        db.session.execute(text('CREATE INDEX IF NOT EXISTS idx_blog_created_at ON blog (created_at DESC)'))
        db.session.execute(text('CREATE INDEX IF NOT EXISTS idx_blog_views ON blog (views DESC)'))
        db.session.execute(text('CREATE INDEX IF NOT EXISTS idx_blog_author ON blog (author_id)'))
        db.session.execute(text('CREATE INDEX IF NOT EXISTS idx_comment_blog ON comment (blog_id)'))
        db.session.execute(text('CREATE INDEX IF NOT EXISTS idx_like_blog ON like (blog_id)'))
        db.session.execute(text('CREATE INDEX IF NOT EXISTS idx_follow_follower ON follow (follower_id)'))
        db.session.execute(text('CREATE INDEX IF NOT EXISTS idx_follow_followed ON follow (followed_id)'))
        db.session.execute(text('CREATE INDEX IF NOT EXISTS idx_hashtag_name ON hashtag (name)'))
        db.session.commit()
    except Exception as e:
        print(f"Warning: Error creating indexes - {str(e)}")
        db.session.rollback()

def downgrade():
    # Drop tables in reverse order to handle dependencies
    try:
        db.session.execute(text('DROP TABLE IF EXISTS blog_hashtags'))
        db.session.execute(text('DROP TABLE IF EXISTS hashtag'))
        db.session.execute(text('DROP TABLE IF EXISTS like'))
        db.session.execute(text('DROP TABLE IF EXISTS comment'))
        db.session.execute(text('DROP TABLE IF EXISTS follow'))
        db.session.execute(text('DROP TABLE IF EXISTS blog'))
        db.session.commit()
    except Exception as e:
        print(f"Warning: Error dropping tables - {str(e)}")
        db.session.rollback() 