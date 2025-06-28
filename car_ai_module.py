import base64
import uuid
import io
import os
import logging
import numpy as np
from datetime import datetime
from PIL import Image
import cv2
import firebase_admin
from firebase_admin import credentials, firestore, storage
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# إعداد السجلات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تهيئة Firebase
def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase-credentials.json")
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'your-bucket-name.appspot.com'  # عدّلي هذا لاسم الباكيت الخاص بك
        })
    return firestore.client(), storage.bucket()

# تحميل النموذج
model = VGG16(weights='imagenet', include_top=False, pooling='avg')

class CarImageProcessor:
    def __init__(self):
        self.db, self.bucket = initialize_firebase()
        self.similarity_threshold = 0.8

    def preprocess_image_from_user(self, image_data):
        try:
            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = self.enhance_image_quality(img)
            return np.array(img)
        except Exception as e:
            logger.error(f"فشل في معالجة الصورة: {e}")
            return None

    def enhance_image_quality(self, img):
        try:
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.warning(f"تحسين الصورة فشل: {e}")
            return img

    def detect_car_in_user_image(self, image_array):
        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
            cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
            if len(cars) > 0:
                x, y, w, h = max(cars, key=lambda x: x[2] * x[3])
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image_array.shape[1] - x, w + 2 * padding)
                h = min(image_array.shape[0] - y, h + 2 * padding)
                return image_array[y:y+h, x:x+w]
            h, w = image_array.shape[:2]
            return image_array[h//4:3*h//4, w//4:3*w//4]
        except Exception as e:
            logger.error(f"فشل في اكتشاف السيارة: {e}")
            return image_array

    def extract_features_from_user_image(self, car_region):
        try:
            img = Image.fromarray(car_region).resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features = model.predict(img_array, verbose=0)
            return features.flatten()
        except Exception as e:
            logger.error(f"خطأ في استخراج المميزات: {e}")
            return None

    def save_user_image_to_storage(self, image_data, user_id, report_id):
        try:
            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes))
            img = self.compress_image(img)

            filename = f"user_reports/{user_id}/{report_id}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
            temp_path = f"temp_{uuid.uuid4().hex}.jpg"
            img.save(temp_path, 'JPEG', quality=85)

            blob = self.bucket.blob(filename)
            blob.upload_from_filename(temp_path)
            blob.make_public()

            if os.path.exists(temp_path):
                os.remove(temp_path)

            return blob.public_url
        except Exception as e:
            logger.error(f"خطأ في رفع الصورة: {e}")
            return None

    def compress_image(self, img, max_size=(800, 600)):
        try:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            return img
        except:
            return img

    def save_user_report_to_firestore(self, user_id, report_data, features, image_url):
        try:
            report_id = str(uuid.uuid4())
            report_doc = {
                'report_id': report_id,
                'user_id': user_id,
                'car_features': features.tolist(),
                'image_url': image_url,
                'report_date': datetime.now(),
                'status': 'active',
                'car_info': report_data.get('car_info', {}),
                'incident_info': report_data.get('incident_info', {}),
                'matching_results': [],
                'last_updated': datetime.now()
            }
            self.db.collection('car_reports').document(report_id).set(report_doc)
            return report_id
        except Exception as e:
            logger.error(f"فشل في حفظ البلاغ: {e}")
            return None

    def search_similar_cars_for_user(self, user_features, user_report_id, incident_type):
        try:
            if incident_type == 'found':
                search_types = ['stolen', 'lost']
            else:
                search_types = ['found']

            query = self.db.collection('car_reports').where('status', '==', 'active').where('incident_info.type', 'in', search_types)
            docs = query.stream()

            matches = []
            for doc in docs:
                data = doc.to_dict()
                if data['report_id'] == user_report_id:
                    continue
                other_features = np.array(data['car_features'])
                similarity = cosine_similarity([user_features], [other_features])[0][0]
                if similarity >= self.similarity_threshold:
                    matches.append({
                        'report_id': data['report_id'],
                        'similarity': float(similarity),
                        'image_url': data['image_url'],
                        'car_info': data['car_info'],
                        'incident_info': data['incident_info']
                    })
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            return matches
        except Exception as e:
            logger.error(f"فشل في البحث عن تطابقات: {e}")
            return []
