import fastapi
import uvicorn
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from pydantic import BaseModel
from typing import List

# --- 1. تحميل النموذج عند بدء التشغيل ---
print("... تحميل نموذج v14 (TabNet) ...")
model = TabNetClassifier()
# (استخدم نفس الاسم من كود التدريب الخاص بك)
model.load_model("tabnet_eurusd_v14_scalper.zip") 
print("... تم تحميل نموذج v14 بنجاح ...")

app = fastapi.FastAPI()

# --- 2. تحديد هيكل البيانات القادمة من MQL5 ---
class FeaturesInput(BaseModel):
    features: List[float] # (نفس الـ 21 ميزة)

# --- 3. إنشاء نقطة النهاية (Endpoint) ---
@app.post("/predict")
async def predict(data: FeaturesInput):
    try:
        # (أ) استلام الـ 21 ميزة كـ list
        features_list = data.features

        # (ب) تحويلها إلى NumPy Array
        # (هذه هي البيانات التي يتوقعها النموذج مباشرة)
        features_np_raw = np.array([features_list]) # (الشكل [1, 21])

        # (ج) طلب التنبؤ (لا حاجة لخطوة المعالجة)
        prediction_tuple = model.predict(features_np_raw)

        # (د) استخراج الإشارة
        signal_raw = prediction_tuple[0][0]
        signal = int(signal_raw)

        print(f"🟢 [v14 Server] تم استلام الميزات. الإشارة = {signal}")

        # (هـ) إرسال الإشارة (0 أو 1) إلى MQL5
        return {"prediction": signal}

    except Exception as e:
        print(f"🔴 [v14 Server] حدث خطأ: {str(e)}")
        raise fastapi.HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():

    return {"message": "خادم v14 (TabNet) يعمل بنجاح!"}


