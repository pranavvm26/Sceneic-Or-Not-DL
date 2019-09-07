import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine

Base = declarative_base()

# define foundation sqlite engine
# Create an engine that stores data in the local directory's
# sqlalchemy_example.db file.
engine = create_engine('sqlite:///rawdata/scenic_or_not.db')


class SoNData(Base):
    __tablename__ = 'data_scenic_or_not'

    # Here we define columns for the table Scenic or Not Data.
    id = Column(Integer, primary_key=True)
    img_latitude = Column(Float)
    img_longitude = Column(Float)
    img_rating_average = Column(Float)
    img_rating_variance = Column(Float)
    img_voting_count = Column(String(250))
    img_source_link = Column(String(500), nullable=False)
    img_link = Column(String(500), nullable=False)
    img_name = Column(String(250), nullable=False)