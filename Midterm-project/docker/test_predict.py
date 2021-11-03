import requests 
url = 'http://localhost:8000/predict' 
customer = {"gender": "male", "customer_type": "loyal_customer", "age":16,"type_of_travel":"business_travel", "flight_distance":311, "inflight_wifi_service":3,"departure/arrival_time_convenient":3,"ease_of_online_booking":3, "gate_location":3, "food_and_drink":5, "online_boarding":5, "seat_comfort":3, "inflight_entertainment":5, "on-board_service":4, "leg_room_service":3, "baggage_handling":1, "checkin_service":1, "inflight_service": 2, "clealiness":5, "arrival_delay_in_minutes":0.0}

response = requests.post(url, json=customer)
result = response.json()
print(result)
