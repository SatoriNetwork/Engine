import grpc
import data_service_pb2
import data_service_pb2_grpc
import pickle
import pandas as pd

def run():
    # Connect to the gRPC server
    channel = grpc.insecure_channel('localhost:50051')
    stub = data_service_pb2_grpc.DataServiceStub(channel)

    # Create a request with the SQL query
    request = data_service_pb2.DataRequest(query="SELECT * FROM my_table")
    response = stub.GetData(request)

    # Deserialize the DataFrame from the response
    df = pickle.loads(response.dataframe)

    # The DataFrame is now ready to use
    print(df)

if __name__ == '__main__':
    run()
