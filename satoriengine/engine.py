from typing import Optional
import time
import threading
from satorilib import logging
from satoriengine.managers.data import DataManager
from satoriengine.managers.model import ModelManager
from satoriengine.view import View


class Engine:

    def __init__(
        self,
        getStart=None,
        data: DataManager = None,
        models: set[ModelManager] = None,
        api: Optional[object] = None,
        view: View = None,
    ):
        '''
        data - a DataManager for the data
        model - a ModelManager for the model
        models - a list of ModelManagers
        '''
        self.getStart = getStart
        self.start = None
        self.data = data
        self.models = models
        self.view = view
        self.api = api

    def out(self, predictions, scores, data=True, model=None):
        ''' old functionality that must be accounted for in new design
        if self.view is not None:
            self.view.print(**(
                {
                    'Predictions:\n':predictions,
                    '\nScores:\n':scores
                } if data else {model.id: 'loading... '}))
        '''
        self.view.print(**(
            {
                'Predictions:\n': predictions,
                '\nScores:\n': scores
            } if data else {model.id: 'loading... '}))

    def updateView(self, predictions, scores):
        '''
        old functionality that must be accounted for in new design
        non-reactive jupyter view
        predictions[model.id] = model.producePrediction()
        scores[model.id] = f'{round(stable, 3)} ({round(test, 3)})'
        inputs[model.id] = model.showFeatureData()
        if first or startingPredictions != predictions:
            first = False
            if self.api is not None:
                self.api.send(model, predictions, scores)
            if self.view is not None:
                self.view.view(model, predictions, scores)
                out()
        out(data=False)
        '''
        if self.view is not None:
            self.view.view(self.data, predictions, scores)
            # out()
        # out(data=False)

    def waitForStartTobeInitialized(self):
        while self.start == None:
            self.start = self.getStart()
            time.sleep(1)

    def run(self):
        ''' Main '''

        def publisher():
            '''
            publishes predictions on demand
            this should probably be broken out into a service
            that creates a stream and a service that publishes...
            '''
            self.data.runPublisher(self.models)

        def subscriber():
            '''
            listens for external updates on subscriptions -
            turn this into a stream rather than a loop -
            triggered from flask app.
            this should probably be broken out into a service
            that subscribes and a service that listens...
            should be on demand
            '''
            self.data.runSubscriber(self.models)

        # # not used yet
        # def scholar():
        #    ''' always looks for external data and compiles it '''
        #    while True:
        #        time.sleep(1)
        #        if not self.start.paused:
        #            self.data.runScholar(self.models)

        def predictor(model: ModelManager):
            ''' produces predictions on demand '''
            model.runPredictor()

        # # not used yet
        # def sync(model: ModelManager):
        #    ''' sync available inputs found and compiled by scholar on demand '''
        #    model.syncAvailableInputs()

        def explorer(model: ModelManager):
            ''' always looks for a better model '''
            while True:
                if not self.start.paused:
                    model.runExplorer()

        def watcher(model: ModelManager):
            ''' for reactive views... '''
            if self.view:
                self.view.listen(model)

        publisher()
        subscriber()
        threads = {}
        # threads['scholar'] = threading.Thread(target=scholar, daemon=True)
        for model in self.models:
            # we have to run this once for each model to complete its initialization
            model.buildStable()
            predictor(model)
            # sync(model)
            if self.view and self.view.isReactive:
                watcher(model)
            self.waitForStartTobeInitialized()
            threads[f'{model.id}.explorer'] = threading.Thread(
                target=explorer, args=[model], daemon=True)
        for thread in threads.values():
            thread.start()


howToRun = '''
# see example notebook
'''
