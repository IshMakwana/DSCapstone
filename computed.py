from lib import *
from sqlalchemy.sql import text

def getManhattanLocIds():
    df = getDF(text(f'select location_id from taxi_zones where location_name=\'Manhattan\''))
    return list(df['location_id'])

manh_locids = getManhattanLocIds()

def getLocationName(location_id):
    # airport location ids = newark=1, JFK Airport,132, LaGuardia Airport,138
    if location_id==1:
        return 'EWR'
    elif location_id==132:
        return 'JFK'
    elif location_id==138:
        return 'LGA'
    elif location_id in manh_locids:
        return 'Manhattan'
    return '-'

def compute_total_amount(row, taxi_type=YELLOW):
    trip_distance = row['trip_distance']
    pickup_datetime = row['pickup_datetime']
    tip_amount = row['tip_amount']
    tolls_amount = row['tolls_amount']
    pickup_location = getLocationName(row['pu_location_id'])
    dropoff_location = getLocationName(row['do_location_id'])
    
    # Initial base fare
    total_amount = 3.00
    
    # Metered fare: 70 cents per 1/5 mile (3.50 per mile)
    total_amount += (trip_distance * 3.50)
    
    # MTA State Surcharge
    total_amount += 0.50
    
    # Improvement Surcharge
    total_amount += 1.00
    
    # Time-based surcharges
    # pickup_time = datetime.datetime.strptime(pickup_datetime, '%Y-%m-%d %H:%M:%S.%f')
    pickup_time = pickup_datetime

    # Overnight surcharge: $1.00 (8pm to 6am)
    if pickup_time.hour >= 20 or pickup_time.hour < 6:
        total_amount += 1.00
    
    # Rush hour surcharge: $2.50 (4pm to 8pm on weekdays, excluding holidays)
    if pickup_time.weekday() < 5 and 16 <= pickup_time.hour < 20:
        total_amount += 2.50
    
    # Airport rules
    if pickup_location in ['LGA', 'JFK', 'EWR'] or dropoff_location in ['LGA', 'JFK', 'EWR']:
        
        # Handle LaGuardia (LGA)
        if pickup_location == 'LGA':
            total_amount += 1.75  # Airport Access Fee for pickup at LGA
            total_amount += 5.00  # Additional $5 surcharge for trips from/to LGA
        
        # Handle JFK Airport
        if pickup_location == 'JFK' or dropoff_location == 'JFK':
            if pickup_location == 'Manhattan' or dropoff_location == 'Manhattan':
                # Flat rate for trips between Manhattan and JFK
                total_amount = 70.00  # Flat rate
                total_amount += 0.50  # MTA Surcharge
                total_amount += 1.00  # Improvement Surcharge
                
                # Rush hour surcharge: $5.00 (4pm to 8pm weekdays)
                if pickup_time.weekday() < 5 and 16 <= pickup_time.hour < 20:
                    total_amount += 2.50
                
                # Set on-screen rate to "Rate #2 - JFK Airport"
                # print("Rate #2 - JFK Airport")
        
        # Handle Newark (EWR)
        if pickup_location == 'EWR' or dropoff_location == 'EWR':
            total_amount += 20.00  # Newark Surcharge
            total_amount += tolls_amount  # Include tolls
            
            # Set on-screen rate to "Rate #3 - Newark Airport"
            # print("Rate #3 - Newark Airport")
    
    # Congestion surcharge for trips in Manhattan south of 96th Street
    # if is_shared_ride:
    #     total_amount += 0.75  # Shared ride
    if taxi_type == GREEN:
        total_amount += 2.75  # Green taxi/FHV
    else:
        total_amount += 2.50  # Yellow taxi

    # Add tips and tolls
    total_amount += tip_amount + tolls_amount
    
    return total_amount

def compute_fare_amount(row, taxi_type=YELLOW):
    trip_distance = row['trip_distance']
    pickup_datetime = row['pickup_datetime']
    tip_amount = row['tip_amount']
    tolls_amount = row['tolls_amount']
    pickup_location = getLocationName(row['pu_location_id'])
    dropoff_location = getLocationName(row['do_location_id'])
    
    # Initial base fare
    fare_amount = 3.00
    
    # Metered fare: 70 cents per 1/5 mile (3.50 per mile)
    fare_amount += (trip_distance * 3.50)
    
    # MTA State Surcharge
    fare_amount += 0.50
    
    # Improvement Surcharge
    fare_amount += 1.00
    
    # Time-based surcharges
    # pickup_time = datetime.datetime.strptime(pickup_datetime, '%Y-%m-%d %H:%M:%S.%f')
    pickup_time = pickup_datetime

    # Overnight surcharge: $1.00 (8pm to 6am)
    if pickup_time.hour >= 20 or pickup_time.hour < 6:
        fare_amount += 1.00
    
    # Rush hour surcharge: $2.50 (4pm to 8pm on weekdays, excluding holidays)
    if pickup_time.weekday() < 5 and 16 <= pickup_time.hour < 20:
        fare_amount += 2.50
    
    # Airport rules
    if pickup_location in ['LGA', 'JFK', 'EWR'] or dropoff_location in ['LGA', 'JFK', 'EWR']:
        
        # Handle LaGuardia (LGA)
        if pickup_location == 'LGA':
            fare_amount += 1.75  # Airport Access Fee for pickup at LGA
            fare_amount += 5.00  # Additional $5 surcharge for trips from/to LGA
        
        # Handle JFK Airport
        if pickup_location == 'JFK' or dropoff_location == 'JFK':
            if pickup_location == 'Manhattan' or dropoff_location == 'Manhattan':
                # Flat rate for trips between Manhattan and JFK
                fare_amount = 70.00  # Flat rate
                fare_amount += 0.50  # MTA Surcharge
                fare_amount += 1.00  # Improvement Surcharge
                
                # Rush hour surcharge: $5.00 (4pm to 8pm weekdays)
                if pickup_time.weekday() < 5 and 16 <= pickup_time.hour < 20:
                    fare_amount += 2.50
                
                # Set on-screen rate to "Rate #2 - JFK Airport"
                # print("Rate #2 - JFK Airport")
        
        # Handle Newark (EWR)
        if pickup_location == 'EWR' or dropoff_location == 'EWR':
            fare_amount += 20.00  # Newark Surcharge
            fare_amount += tolls_amount  # Include tolls
            
            # Set on-screen rate to "Rate #3 - Newark Airport"
            # print("Rate #3 - Newark Airport")
    
    # Congestion surcharge for trips in Manhattan south of 96th Street
    # if is_shared_ride:
    #     fare_amount += 0.75  # Shared ride
    if taxi_type == GREEN:
        fare_amount += 2.75  # Green taxi/FHV
    else:
        fare_amount += 2.50  # Yellow taxi
    
    return fare_amount