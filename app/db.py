from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker

from app.helpers.config import DATABASE_URL

engine = create_engine(DATABASE_URL)

Session = scoped_session(sessionmaker(bind=engine))
