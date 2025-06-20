'''
basic structure of the Veda Engine:
    AIEngine
    | (has many)
    StreamModel
    | (manages and explores models wrapped in...)
    ModelAdapter
    | (wraps...)
    pipelines and algorithms, etc.

interpolation:
1. can this model use Nan interpolation
2. if not fill in the missing data with the model itself (for target variable)
3. for date features fill in
4. for other features - nan if possible
5. for other features - if not possible, fill in with step data
'''
# from satoriengine.veda import engine
