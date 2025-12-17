# /workspace/Project/models.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

# Define the SQLAlchemy instance globally
db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    fencers = db.relationship('Fencer', backref='user', lazy=True)
    uploads = db.relationship('Upload', backref='user', lazy=True)

class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    # For backward compatibility, keep video_path for single video uploads
    video_path = db.Column(db.String(255), nullable=True)  # Changed to nullable for multi-video
    match_datetime = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(50), nullable=False)
    detection_image_path = db.Column(db.String(255))
    cross_bout_analysis_path = db.Column(db.String(255))
    left_fencer_id = db.Column(db.Integer, db.ForeignKey('fencer.id'))
    right_fencer_id = db.Column(db.Integer, db.ForeignKey('fencer.id'))
    selected_indexes = db.Column(db.String(50))
    weapon_type = db.Column(db.String(20), nullable=False, default='saber')  # 'saber', 'foil', or 'epee'
    total_bouts = db.Column(db.Integer)
    output_video_path = db.Column(db.String(255))
    csv_dir = db.Column(db.String(255))
    bouts_analyzed = db.Column(db.Integer, default=0)
    # Multi-video support
    is_multi_video = db.Column(db.Boolean, default=False)
    match_title = db.Column(db.String(255))  # User-defined title for multi-video match
    bouts = db.relationship('Bout', backref='upload', lazy=True)
    videos = db.relationship('UploadVideo', backref='upload', lazy=True, order_by='UploadVideo.sequence_order')

class UploadVideo(db.Model):
    """Individual video within a multi-video upload"""
    id = db.Column(db.Integer, primary_key=True)
    upload_id = db.Column(db.Integer, db.ForeignKey('upload.id'), nullable=False)
    video_path = db.Column(db.String(255), nullable=False)
    sequence_order = db.Column(db.Integer, nullable=False)  # Order in the match sequence
    selected_indexes = db.Column(db.String(50))  # Tracking indexes for this video
    status = db.Column(db.String(50), default='pending')  # 'pending', 'processing', 'completed', 'error'
    detection_image_path = db.Column(db.String(255))
    total_bouts = db.Column(db.Integer, default=0)
    bouts_offset = db.Column(db.Integer, default=0)  # Starting bout number for this video in the overall sequence
    
class Fencer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(150), nullable=False)
    uploads_as_left = db.relationship('Upload', foreign_keys=[Upload.left_fencer_id], backref=db.backref('left_fencer', lazy=True))
    uploads_as_right = db.relationship('Upload', foreign_keys=[Upload.right_fencer_id], backref=db.backref('right_fencer', lazy=True))

class Bout(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    upload_id = db.Column(db.Integer, db.ForeignKey('upload.id'), nullable=False)
    upload_video_id = db.Column(db.Integer, db.ForeignKey('upload_video.id'), nullable=True)  # Which video this bout came from
    match_idx = db.Column(db.Integer, nullable=False)
    start_frame = db.Column(db.Integer, nullable=False)
    end_frame = db.Column(db.Integer, nullable=False)
    video_path = db.Column(db.String(255), nullable=True)
    extended_video_path = db.Column(db.String(255), nullable=True)  # Video with 1s padding on each side for display
    result = db.Column(db.String(10), nullable=True)  # 'left', 'right', or 'skip'
    upload_video = db.relationship('UploadVideo', backref='bouts', lazy=True)

class HolisticAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fencer_id = db.Column(db.Integer, db.ForeignKey('fencer.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    report_path = db.Column(db.String(255), nullable=True)
    status = db.Column(db.String(50), default='Pending')
    fencer = db.relationship('Fencer', backref='holistic_analyses', uselist=False)
    user = db.relationship('User', backref='holistic_analyses')

class Tag(db.Model):
    __tablename__ = 'tag'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)

    def __repr__(self):
        return f'<Tag {self.name}>'

class BoutTag(db.Model):
    __tablename__ = 'bout_tag'
    id = db.Column(db.Integer, primary_key=True)
    bout_id = db.Column(db.Integer, db.ForeignKey('bout.id'), nullable=False)
    tag_id = db.Column(db.Integer, db.ForeignKey('tag.id'), nullable=False)
    fencer_side = db.Column(db.String(5), nullable=False)  # 'left' or 'right'

    # Add relationships to easily access related objects
    bout = db.relationship('Bout', backref=db.backref('tags', lazy='dynamic', cascade="all, delete-orphan"))
    tag = db.relationship('Tag', backref=db.backref('bout_tags', lazy='dynamic'))

    def __repr__(self):
        return f'<BoutTag bout_id={self.bout_id} tag={self.tag.name} side={self.fencer_side}>'

class VideoAnalysis(db.Model):
    """Store AI-generated video analysis for uploads"""
    __tablename__ = 'video_analysis'
    id = db.Column(db.Integer, primary_key=True)
    upload_id = db.Column(db.Integer, db.ForeignKey('upload.id'), nullable=False, unique=True)
    
    # Overall performance analysis (JSON stored as text)
    left_overall_analysis = db.Column(db.Text, nullable=True)  # JSON string
    right_overall_analysis = db.Column(db.Text, nullable=True)  # JSON string
    
    # Category performance analysis (JSON stored as text)
    left_category_analysis = db.Column(db.Text, nullable=True)  # JSON string with in_box, attack, defense
    right_category_analysis = db.Column(db.Text, nullable=True)  # JSON string with in_box, attack, defense
    
    # Loss analysis (JSON stored as text)  
    loss_analysis = db.Column(db.Text, nullable=True)  # JSON string with grouped loss reasons
    
    # Performance metrics and detailed analysis
    detailed_analysis = db.Column(db.Text, nullable=True)  # JSON string with mirror bar chart data
    
    # Metadata
    generated_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    status = db.Column(db.String(20), default='pending')  # 'pending', 'completed', 'error'
    error_message = db.Column(db.Text, nullable=True)
    
    # Relationship
    upload = db.relationship('Upload', backref=db.backref('video_analysis', uselist=False, cascade="all, delete-orphan"))
    
    def __repr__(self):
        return f'<VideoAnalysis upload_id={self.upload_id} status={self.status}>'
