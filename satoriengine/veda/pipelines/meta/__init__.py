'''
the idea:
    chronos + lightgbm + its own metamodel on hyperparameter search

model it after xgb_chronos where we produce a prediction using untrained chronos
as a feature input to lightgbm which it can use to learn how well and in what
context chronos has useful predictive power.

Then each time you make a new hyper parameter save it, along with it's score
over time. the score over time is the number of observations in the past it's
predicted well (above some threshold). the datastructure could look like this:

```
class ParameterSet (unique object)
  ...
hyperparameters = {parameterSet: [score, historic score, historic score...]}
```

we score them and then we train an embedding (which represents a semantic space)
on the parameterSet and the score. we also train a regression model so that when
we generate a new set of hyperparameters we can check what score is most likely
implied by the semantic space. Then we can sample around clusters of high
scoring hyper parameters to get good candidates.
'''
