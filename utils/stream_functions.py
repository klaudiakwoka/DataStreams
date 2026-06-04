def warsaw_stream(df):
    for row in df:
        x = dict(row)
        y = x.get('travel_time_min')
        x.pop('travel_time_min',None)
        yield x, y

def airline_stream(df):
    for row in df:
        x = dict(row)
        y = x.get('ArrDelay')
        x.pop('ArrDelay',None)
        yield x, y

def taxi_stream(df):
    for row in df:
        x = dict(row)
        y = x.get('trip_duration')
        y=round(float(y/60),2)
        x.pop('trip_duration',None)
        yield x, y