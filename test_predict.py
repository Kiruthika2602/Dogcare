# test_predict_samples.py
from src.models.predictor import predict_weight_only, predict_calorie_only, predict_activity_only

samples = {
    "indian_spitz": {
        "breed": "Indian Spitz", "sex": "F", "age_months": 26, "height_cm": 34.0, "length_cm": 48.0,
        "avg_daily_play_min": 35.0, "avg_daily_walk_min": 50.0, "sleep_hours": 11.0, "weight_kg": None
    },
    "golden": {
        "breed": "Golden Retriever", "sex": "M", "age_months": 18, "height_cm": 60.0, "length_cm": 70.0,
        "avg_daily_play_min": 45.0, "avg_daily_walk_min": 60.0, "sleep_hours": 10.0, "weight_kg": 28.0
    },
    "pug": {
        "breed": "Pug", "sex": "F", "age_months": 40, "height_cm": 0.0, "length_cm": 0.0,
        "avg_daily_play_min": 10.0, "avg_daily_walk_min": 15.0, "sleep_hours": 14.5, "weight_kg": None
    }
}

for name, s in samples.items():
    print("=== SAMPLE:", name)
    print("Weight only ->", predict_weight_only(s))
    try:
        print("Calorie only ->", predict_calorie_only(s))
    except Exception as e:
        print("Calorie failed:", e)
    print("Activity only ->", predict_activity_only(s))
    print()
