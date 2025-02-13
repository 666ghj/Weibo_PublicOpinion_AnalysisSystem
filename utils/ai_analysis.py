from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class AIAnalysis(Base):
    __tablename__ = 'ai_analysis'
    
    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, nullable=False)
    sentiment = Column(String(10), nullable=False)
    sentiment_score = Column(Float, nullable=False)
    keywords = Column(JSON, nullable=False)
    key_points = Column(Text, nullable=False)
    influence_analysis = Column(Text, nullable=False)
    risk_level = Column(String(10), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    
    def to_dict(self):
        return {
            'id': self.id,
            'message_id': self.message_id,
            'sentiment': self.sentiment,
            'sentiment_score': f"{self.sentiment_score:.2%}",
            'keywords': self.keywords,
            'key_points': self.key_points,
            'influence': self.influence_analysis,
            'risk_level': self.risk_level,
            'analysis_time': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        } 