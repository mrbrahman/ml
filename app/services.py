# Facade pattern: expose managers through a single import point
from app.core.face_recognition import manager as faces
from app.core.image_analysis import manager as images, search
from app.core.face_recognition import storage, clustering
from app.core.face_recognition.annotator import create_enriched_image
from app.core import model_loader
