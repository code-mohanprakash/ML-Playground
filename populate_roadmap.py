from app import app, db
from models.roadmap import CareerPath, Milestone, Skill, Resource
from datetime import datetime

def populate_roadmap_data():
    """Populate the database with roadmap data."""
    with app.app_context():
        # Check if data already exists
        if CareerPath.query.count() > 0:
            print("Roadmap data already exists. Skipping...")
            return
        
        print("Populating roadmap data...")
        
        # Create career paths
        data_analyst = CareerPath(
            name='data-analyst',
            description='Learn to analyze data and create meaningful insights for business decisions.',
            icon='chart-line'
        )
        
        data_scientist = CareerPath(
            name='data-scientist',
            description='Master machine learning and statistical analysis to solve complex problems.',
            icon='brain'
        )
        
        ml_engineer = CareerPath(
            name='ml-engineer',
            description='Build and deploy machine learning systems at scale.',
            icon='cogs'
        )
        
        ai_engineer = CareerPath(
            name='ai-engineer',
            description='Develop cutting-edge AI solutions and deep learning models.',
            icon='robot'
        )
        
        research_scientist = CareerPath(
            name='research-scientist',
            description='Conduct research and develop novel ML/AI algorithms.',
            icon='microscope'
        )
        
        db.session.add_all([data_analyst, data_scientist, ml_engineer, ai_engineer, research_scientist])
        db.session.commit()
        
        # Create milestones for Data Analyst path
        da_milestones = [
            Milestone(
                title='Foundations of Data Analysis',
                description='Learn the fundamental concepts and tools for data analysis.',
                order=1,
                duration='4-6 weeks',
                difficulty='Beginner',
                career_path_id=data_analyst.id
            ),
            Milestone(
                title='Data Manipulation & Cleaning',
                description='Master techniques for cleaning and transforming data for analysis.',
                order=2,
                duration='4-6 weeks',
                difficulty='Beginner',
                career_path_id=data_analyst.id
            ),
            Milestone(
                title='Exploratory Data Analysis',
                description='Learn to explore and visualize data to uncover patterns and insights.',
                order=3,
                duration='6-8 weeks',
                difficulty='Intermediate',
                career_path_id=data_analyst.id
            ),
            Milestone(
                title='Statistical Analysis',
                description='Apply statistical methods to analyze data and test hypotheses.',
                order=4,
                duration='6-8 weeks',
                difficulty='Intermediate',
                career_path_id=data_analyst.id
            ),
            Milestone(
                title='Data Visualization & Storytelling',
                description='Create compelling visualizations and narratives from data.',
                order=5,
                duration='4-6 weeks',
                difficulty='Intermediate',
                career_path_id=data_analyst.id
            ),
            Milestone(
                title='Business Intelligence Tools',
                description='Master BI tools like Tableau, Power BI, and SQL for business analytics.',
                order=6,
                duration='6-8 weeks',
                difficulty='Advanced',
                career_path_id=data_analyst.id
            )
        ]
        
        db.session.add_all(da_milestones)
        db.session.commit()
        
        # Create milestones for Data Scientist path
        ds_milestones = [
            Milestone(
                title='Foundations of Data Science',
                description='Learn the fundamental concepts and tools for data science.',
                order=1,
                duration='4-6 weeks',
                difficulty='Beginner',
                career_path_id=data_scientist.id
            ),
            Milestone(
                title='Programming for Data Science',
                description='Master Python and essential libraries for data manipulation and analysis.',
                order=2,
                duration='6-8 weeks',
                difficulty='Beginner',
                career_path_id=data_scientist.id
            ),
            Milestone(
                title='Machine Learning Fundamentals',
                description='Learn the core concepts and algorithms of machine learning.',
                order=3,
                duration='8-10 weeks',
                difficulty='Intermediate',
                career_path_id=data_scientist.id
            ),
            Milestone(
                title='Advanced Machine Learning',
                description='Master advanced ML techniques and ensemble methods.',
                order=4,
                duration='8-10 weeks',
                difficulty='Advanced',
                career_path_id=data_scientist.id
            ),
            Milestone(
                title='Deep Learning',
                description='Learn neural networks and deep learning architectures.',
                order=5,
                duration='8-10 weeks',
                difficulty='Advanced',
                career_path_id=data_scientist.id
            ),
            Milestone(
                title='Natural Language Processing',
                description='Process and analyze text data using NLP techniques.',
                order=6,
                duration='6-8 weeks',
                difficulty='Advanced',
                career_path_id=data_scientist.id
            ),
            Milestone(
                title='Data Science in Production',
                description='Deploy and monitor machine learning models in production.',
                order=7,
                duration='6-8 weeks',
                difficulty='Advanced',
                career_path_id=data_scientist.id
            )
        ]
        
        db.session.add_all(ds_milestones)
        db.session.commit()
        
        # Create milestones for ML Engineer path
        ml_milestones = [
            Milestone(
                title='Software Engineering Fundamentals',
                description='Learn software development principles and best practices.',
                order=1,
                duration='6-8 weeks',
                difficulty='Beginner',
                career_path_id=ml_engineer.id
            ),
            Milestone(
                title='Machine Learning Foundations',
                description='Master the core concepts and algorithms of machine learning.',
                order=2,
                duration='8-10 weeks',
                difficulty='Intermediate',
                career_path_id=ml_engineer.id
            ),
            Milestone(
                title='ML Systems Design',
                description='Design scalable and efficient machine learning systems.',
                order=3,
                duration='6-8 weeks',
                difficulty='Intermediate',
                career_path_id=ml_engineer.id
            ),
            Milestone(
                title='ML Model Deployment',
                description='Deploy ML models to production environments.',
                order=4,
                duration='6-8 weeks',
                difficulty='Advanced',
                career_path_id=ml_engineer.id
            ),
            Milestone(
                title='MLOps & Monitoring',
                description='Implement ML operations and monitoring systems.',
                order=5,
                duration='6-8 weeks',
                difficulty='Advanced',
                career_path_id=ml_engineer.id
            ),
            Milestone(
                title='Distributed ML Systems',
                description='Build and optimize distributed machine learning systems.',
                order=6,
                duration='8-10 weeks',
                difficulty='Expert',
                career_path_id=ml_engineer.id
            )
        ]
        
        db.session.add_all(ml_milestones)
        db.session.commit()
        
        # Add skills and resources for the first milestone of Data Analyst
        da_first_milestone = da_milestones[0]
        
        # Skills
        da_skills = [
            Skill(
                name='Excel for Data Analysis',
                description='Use Excel for data manipulation, analysis, and visualization.',
                category='Tools',
                milestone_id=da_first_milestone.id
            ),
            Skill(
                name='SQL Basics',
                description='Write SQL queries to extract and manipulate data from databases.',
                category='Technical',
                milestone_id=da_first_milestone.id
            ),
            Skill(
                name='Data Types & Structures',
                description='Understand different data types and structures used in data analysis.',
                category='Technical',
                milestone_id=da_first_milestone.id
            ),
            Skill(
                name='Basic Statistics',
                description='Apply descriptive statistics to summarize and understand data.',
                category='Technical',
                milestone_id=da_first_milestone.id
            )
        ]
        
        db.session.add_all(da_skills)
        db.session.commit()
        
        # Resources
        da_resources = [
            Resource(
                title='Excel Data Analysis: Pivot Tables',
                url='https://www.coursera.org/learn/excel-data-analysis-pivot-tables',
                type='Course',
                provider='Coursera',
                is_free=False,
                milestone_id=da_first_milestone.id
            ),
            Resource(
                title='SQL for Data Analysis',
                url='https://www.udacity.com/course/sql-for-data-analysis--ud198',
                type='Course',
                provider='Udacity',
                is_free=True,
                milestone_id=da_first_milestone.id
            ),
            Resource(
                title='Statistics and Probability',
                url='https://www.khanacademy.org/math/statistics-probability',
                type='Course',
                provider='Khan Academy',
                is_free=True,
                milestone_id=da_first_milestone.id
            ),
            Resource(
                title='Data Analysis with Python',
                url='https://www.freecodecamp.org/learn/data-analysis-with-python/',
                type='Course',
                provider='freeCodeCamp',
                is_free=True,
                milestone_id=da_first_milestone.id
            )
        ]
        
        db.session.add_all(da_resources)
        db.session.commit()
        
        print("Roadmap data populated successfully!")

if __name__ == "__main__":
    populate_roadmap_data() 