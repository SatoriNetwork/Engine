# server.py
import grpc
from concurrent import futures
import data_service_pb2
import data_service_pb2_grpc
import pickle
import pandas as pd
import sqlite3
from datetime import datetime

class DataService(data_service_pb2_grpc.DataServiceServicer):
    def __init__(self):
        # Initialize database and create table
        self.setup_database()

    def setup_database(self):
        conn = sqlite3.connect('my_database.db')
        cursor = conn.cursor()
        
        # Create table with timestamp, value, and hash columns
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS my_table (
            timestamp TIMESTAMP NOT NULL,
            value FLOAT NOT NULL,
            hash_value TEXT NOT NULL
        )
        ''')
        
        # Insert sample data matching your format
        sample_data = [
            ('2024-10-24 07:10:21.089727', 984.7420666666667, '8986613eb327edc2'),
            # Adding a few more sample entries
            ('2024-10-24 07:10:22.089727', 985.1234567890123, '9986613eb327edc3'),
            ('2024-10-24 07:10:23.089727', 983.9876543210987, '7986613eb327edc4')
        ]
        
        cursor.execute('DELETE FROM my_table')  # Clear existing data
        cursor.executemany(
            'INSERT INTO my_table (timestamp, value, hash_value) VALUES (?, ?, ?)', 
            sample_data
        )
        
        # Create an index on timestamp for better query performance
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_timestamp 
        ON my_table (timestamp)
        ''')
        
        conn.commit()
        conn.close()

    def GetData(self, request, context):
        # Connect to database
        conn = sqlite3.connect('my_database.db')
        
        try:
            # Configure SQLite to return timestamps as strings
            conn.execute("PRAGMA busy_timeout = 5000")  # Set timeout to 5s
            
            # Execute query and convert to DataFrame
            df = pd.read_sql_query(request.query, conn)
            
            # Ensure timestamp is in the correct format
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Serialize DataFrame
            serialized_df = pickle.dumps(df)
            
            return data_service_pb2.DataResponse(dataframe=serialized_df)
        
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise
        
        finally:
            conn.close()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    data_service_pb2_grpc.add_DataServiceServicer_to_server(
        DataService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
