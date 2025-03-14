from typing import Union
import os
import json
import copy
import random
import asyncio
import warnings
import threading
import numpy as np
import pandas as pd
from satorilib.concepts import Observation
from satorilib.logging import INFO, setup, debug, info, warning, error
from satorilib.utils.system import getProcessorCount
from satorilib.utils.time import datetimeToTimestamp, now
from satorilib.datamanager import DataClient, DataServerApi, DataClientApi, PeerInfo, Message, Subscription
from satorilib.wallet.evrmore.identity import EvrmoreIdentity
from satorilib.pubsub import SatoriPubSubConn
from satoriengine.veda import config
from satoriengine.veda.data import StreamForecast, validate_single_entry
from satoriengine.veda.adapters import ModelAdapter, StarterAdapter, XgbAdapter, XgbChronosAdapter

warnings.filterwarnings('ignore')
setup(level=INFO)



class Engine:

    @classmethod
    async def create(cls) -> 'Engine':
        engine = cls()
        await engine.initialize()
        return engine

    def __init__(self):
        self.streamModels: dict[str, StreamModel] = {}
        self.subscriptions: dict[str, PeerInfo] = {}
        self.publications: dict[str, PeerInfo] = {}
        self.dataServerIp: str = ''
        self.dataServerPort: Union[int, None] = None
        self.dataClient: Union[DataClient, None] = None
        self.paused: bool = False
        self.threads: list[threading.Thread] = []
        self.identity: EvrmoreIdentity = EvrmoreIdentity('/Satori/Neuron/wallet/wallet.yaml')
        # TODO: handle the server - doesn't the neuron send our predictions to the central server to be scored? if so we don't need this here.
        #self.server: SatoriServerClient = None
        self.sub: SatoriPubSubConn = None
        # TOOD: cleanup - maybe this should be passed in and key'ed off ENV like was done before?
        self.urlPubsubs={
                # 'local': ['ws://192.168.0.10:24603'],
                'local': ['ws://pubsub1.satorinet.io:24603', 'ws://pubsub5.satorinet.io:24603', 'ws://pubsub6.satorinet.io:24603'],
                'dev': ['ws://localhost:24603'],
                'test': ['ws://test.satorinet.io:24603'],
                'prod': ['ws://pubsub1.satorinet.io:24603', 'ws://pubsub5.satorinet.io:24603', 'ws://pubsub6.satorinet.io:24603']}['prod']
        self.transferProtocol: Union[str, None] = None
        self.transferProtocolPayload: Union[dict, None] = None


    ## TODO: fix addStream to work with the new way init looks, not the old way:
    #
    #def __init__(self, streams: list[Stream], pubStreams: list[Stream]):
    #    self.streams = streams
    #    self.pubStreams = pubStreams
    #    self.streamModels: Dict[StreamId, StreamModel] = {}
    #    self.newObservation: BehaviorSubject = BehaviorSubject(None)
    #    self.predictionProduced: BehaviorSubject = BehaviorSubject(None)
    #    self.setupSubscriptions()
    #    self.initializeModels()
    #    self.paused: bool = False
    #    self.threads: list[threading.Thread] = []
    #
    #def addStream(self, stream: Stream, pubStream: Stream):
    #    ''' add streams to a running engine '''
    #    # don't duplicate effort
    #    if stream.streamId.uuid in [s.streamId.uuid for s in self.streams]:
    #        return
    #    self.streams.append(stream)
    #    self.pubStreams.append(pubStream)
    #    self.streamModels[stream.streamId] = StreamModel(
    #        streamId=stream.streamId,
    #        predictionStreamId=pubStream.streamId,
    #        predictionProduced=self.predictionProduced)
    #    self.streamModels[stream.streamId].chooseAdapter(inplace=True)
    #    self.streamModels[stream.streamId].run_forever()

    def proactivelyConnectToSubscribers(self, subscribers: list[dict]):
        '''
        # todo: move this to the StreamModel.
        establish connections to subscriber data servers and provide
        observations on the datastreams they subscribe to.
        subscribers should look lik this:
        [{
            'subscription_ds_uuid': '588d9361-64cc-5fb7-8a1d-9395d70f622d',
            'ip': '70.30.39.94',
            'ip_ts': Timestamp('2025-03-02 21:03:47.628513'),
            'author': '038cc6020b9ef6a3437c032a07a3b9ef7c32101c7d7baf8d567e1d15ef4d68374d',
            'prediction_ds_uuid': '1a854c56-4093-5182-b17a-99dd3bbe9e79'},
            ...
        ]
        '''
        # we need to loop through the subscribers to compile a list of subscription_ds_uuids for each unique author-ip combo.
        # we can use the 'author' to validate their identity
        # we can use the 'ip' to connect to their data server
        # we can use the 'subscription_ds_uuid' to tell them what datastreams we'll be sending to them


    def subConnect(self, key: str):
        """establish a random pubsub connection used only for subscribing"""

        def establishConnection(
            pubkey: str,
            key: str,
            url: str = None,
            onConnect: callable = None,
            onDisconnect: callable = None,
            emergencyRestart: callable = None,
            subscription: bool = True,
        ):
            """establishes a connection to the satori server, returns connection object"""
            from satorineuron.init.start import getStart

            def router(response: str):
                ''' gets observation from pubsub servers '''
                # response:
                # {"topic": "{\"source\": \"satori\", \"author\": \"021bd7999774a59b6d0e40d650c2ed24a49a54bdb0b46c922fd13afe8a4f3e4aeb\", \"stream\": \"coinbaseALGO-USD\", \"target\": \"data.rates.ALGO\"}", "data": "0.23114999999999997"}
                if (
                    response
                    != "failure: error, a minimum 10 seconds between publications per topic."
                ):
                    if response.startswith('{"topic":') or response.startswith('{"data":'):
                        try:
                            obs = Observation.parse(response)
                            info(
                                'received:',
                                f'\n {obs.streamId.cleanId}',
                                f'\n ({obs.value}, {obs.observationTime}, {obs.observationHash})',
                                print=True)
                            streamModel = self.streamModels.get(obs.streamId.uuid)
                            if isinstance(streamModel, StreamModel):
                                def run_async_in_thread():
                                    try:
                                        loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(loop)
                                        loop.run_until_complete(
                                            streamModel.handleSubscriptionMessage(
                                                message=Message({
                                                    'data': obs,
                                                    'status': 'stream/observation'}),
                                                pubSubFlag=True)
                                        )
                                    except Exception as e:
                                        print(f"Exception in async thread: {e}")
                                        import traceback
                                        traceback.print_exc()
                                    finally:
                                        loop.close()
                                thread = threading.Thread(target=run_async_in_thread)
                                thread.daemon = True
                                thread.start()
                        except json.JSONDecodeError:
                            info('received unparsable message:', response, print=True)
                    else:
                        info('received:', response, print=True)

            info(
                'subscribing to:' if subscription else 'publishing to:', url, color='blue')
            return SatoriPubSubConn(
                uid=pubkey,
                router=router if subscription else None,
                payload=key,
                url=url,
                emergencyRestart=emergencyRestart,
                onConnect=onConnect,
                onDisconnect=onDisconnect,
            )
            # payload={
            #    'publisher': ['stream-a'],
            #    'subscriptions': ['stream-b', 'stream-c', 'stream-d']})


        # accept optional data necessary to generate models data and learner


        # TODO: should we even do this?
        #if self.sub is not None:
        #    self.sub.disconnect()
        #    # TODO replace to get this information to the UI somehow.
        #    #self.updateConnectionStatus(
        #    #    connTo=ConnectionTo.pubsub, status=False)
        #    self.sub = None
        signature = self.identity.sign(key)
        self.sub = establishConnection(
            url=random.choice(self.urlPubsubs),
            # url='ws://pubsub3.satorinet.io:24603',
            pubkey=self.identity.publicKey,
            key=signature.decode() + "|" + key,
            emergencyRestart=lambda: print('emergencyRestart not implemented'),
            onConnect=lambda: print('onConnect not implemented'),
            onDisconnect=lambda: print('onDisconnect not implemented'))
            # TODO: tell the UI we disconnected, and reconnected... somehow...
            #onConnect=lambda: self.updateConnectionStatus(
            #    connTo=ConnectionTo.pubsub,
            #    status=True),
            #onDisconnect=lambda: self.updateConnectionStatus(
            #    connTo=ConnectionTo.pubsub,
            #    status=False))

    async def initialize(self):
        await self.connectToDataServer()
        asyncio.create_task(self.stayConnectedForever())
        await self.startService()

    async def startService(self):
        await self.getPubSubInfo()
        await self.initializeModels()
        # await asyncio.Event().wait()

    def addStream(self, stream: Stream, pubStream: Stream):
        ''' add streams to a running engine '''
        # don't duplicate effort
        if stream.streamId.uuid in [s.streamId.uuid for s in self.streams]:
            return
        self.streams.append(stream)
        self.pubstreams.append(pubStream)
        self.streamModels[stream.streamId] = StreamModel(
            streamId=stream.streamId,
            predictionStreamId=pubStream.streamId,
            predictionProduced=self.predictionProduced)
        self.streamModels[stream.streamId].chooseAdapter(inplace=True)
        self.streamModels[stream.streamId].run_forever()

    def pause(self, force: bool = False):
        if force:
            self.paused = True
        for streamModel in self.streamModels.values():
            streamModel.pause()

    def resume(self, force: bool = False):
        if force:
            self.paused = False
        if not self.paused:
            for streamModel in self.streamModels.values():
                streamModel.resume()

    @property
    def isConnectedToServer(self):
        if hasattr(self, 'dataClient') and self.dataClient is not None:
            return self.dataClient.isConnected()
        return False

    async def connectToDataServer(self):
        ''' connect to server, retry if failed '''

        async def authenticate() -> bool:
            response = await self.dataClient.authenticate(islocal='engine')
            if response.status == DataServerApi.statusSuccess.value:
                info("Local Engine successfully connected to Server Ip at :", self.dataServerIp, color="green")
                return True
            return False

        async def initiateServerConnection() -> bool:
            ''' local engine client authorization '''
            self.dataClient = DataClient(self.dataServerIp, self.dataServerPort, identity=self.identity)
            return await authenticate()

        waitingPeriod = 10
        while not self.isConnectedToServer:
            try:
                self.dataServerIp = config.get().get('server ip', '0.0.0.0')
                self.dataServerPort = int(config.get().get('server port', 24600))
                if await initiateServerConnection():
                    return True
            except Exception as e:
                warning(f'Failed to find a valid Server Ip, retrying in {waitingPeriod}')
                await asyncio.sleep(waitingPeriod)

    async def getPubSubInfo(self):
        ''' gets the relation info between pub-sub streams '''
        waitingPeriod = 10
        while not self.subscriptions and self.isConnectedToServer:
            try:
                pubSubResponse: Message = await self.dataClient.getPubsubMap()
                self.transferProtocol = pubSubResponse.streamInfo.get('transferProtocol')
                self.transferProtocolPayload = pubSubResponse.streamInfo.get('transferProtocolPayload')
                pubSubMapping = pubSubResponse.streamInfo.get('pubSubMapping')
                if pubSubResponse.status == DataServerApi.statusSuccess.value and pubSubMapping:
                    for sub_uuid, data in pubSubMapping.items():
                        # TODO : deal with supportive streams, ( data['supportiveUuid'] )
                        self.subscriptions[sub_uuid] = PeerInfo(data['dataStreamSubscribers'], data['dataStreamPublishers'])
                        self.publications[data['publicationUuid']] = PeerInfo(data['predictiveStreamSubscribers'], data['predictiveStreamPublishers'])
                    if self.subscriptions:
                        info(pubSubResponse.senderMsg, color='green')
                else:
                    raise Exception
                if self.transferProtocol == 'pubsub':
                    self.subConnect(key=self.transferProtocolPayload)
                    return
                # if self.transferProtocol == 'p2p-proactive':
                #     self.proactivelyConnectToSubscribers(subscribers=self.transferProtocolPayload)
                #     return
            except Exception:
                warning(f"Failed to fetch pub-sub info, waiting for {waitingPeriod} seconds")
                await asyncio.sleep(waitingPeriod)

    async def stayConnectedForever(self):
        ''' alternative to await asyncio.Event().wait() '''
        while True:
            await asyncio.sleep(10)
            if not self.isConnectedToServer:
                await self.connectToDataServer()
                await self.startService()

    async def initializeModels(self):
        info(f'Transfer protocol : {self.transferProtocol}', color='green')
        for subUuid, pubUuid in zip(self.subscriptions.keys(), self.publications.keys()):
            peers = self.subscriptions[subUuid]
            try:
                self.streamModels[subUuid] = await StreamModel.create(
                    streamUuid=subUuid,
                    predictionStreamUuid=pubUuid,
                    peerInfo=peers,
                    dataClient=self.dataClient,
                    pauseAll=self.pause,
                    resumeAll=self.resume,
                    transferProtocol=self.transferProtocol)
            except Exception as e:
                error(e)
            self.streamModels[subUuid].chooseAdapter(inplace=True)
            self.streamModels[subUuid].run_forever()

    def cleanupThreads(self):
        for thread in self.threads:
            if not thread.is_alive():
                self.threads.remove(thread)
        debug(f'prediction thread count: {len(self.threads)}')


class StreamModel:

    @classmethod
    async def create(
        cls,
        streamUuid: str,
        predictionStreamUuid: str,
        peerInfo: PeerInfo,
        dataClient: DataClient,
        pauseAll:callable,
        resumeAll:callable,
        transferProtocol: str
    ):
        streamModel = cls(
            streamUuid,
            predictionStreamUuid,
            peerInfo,
            dataClient,
            pauseAll,
            resumeAll,
            transferProtocol
        )
        await streamModel.initialize()
        return streamModel

    def __init__(
        self,
        streamUuid: str,
        predictionStreamUuid: str,
        peerInfo: PeerInfo,
        dataClient: DataClient,
        pauseAll:callable,
        resumeAll:callable,
        transferProtocol: str
    ):
        self.cpu = getProcessorCount()
        self.pauseAll = pauseAll
        self.resumeAll = resumeAll
        self.preferredAdapters: list[ModelAdapter] = [StarterAdapter, XgbAdapter, XgbChronosAdapter]# SKAdapter #model[0] issue
        self.defaultAdapters: list[ModelAdapter] = [XgbAdapter, XgbAdapter, StarterAdapter]
        self.failedAdapters = []
        self.thread: threading.Thread = None
        self.streamUuid: str = streamUuid
        self.predictionStreamUuid: str = predictionStreamUuid
        self.peerInfo: PeerInfo = peerInfo
        self.dataClient: DataClient = dataClient
        self.rng = np.random.default_rng(37)
        self.publisherHost = None
        self.transferProtocol: str = transferProtocol

    async def initialize(self):
        self.data: pd.DataFrame = await self.loadData()
        self.adapter: ModelAdapter = self.chooseAdapter()
        self.pilot: ModelAdapter = self.adapter(uid=self.streamUuid)
        self.pilot.load(self.modelPath())
        self.stable: ModelAdapter = copy.deepcopy(self.pilot)
        self.paused: bool = False
        debug(f'AI Engine: stream id {self.streamUuid} using {self.adapter.__name__}', color='teal')

    async def p2pInit(self):
        await self.connectToPeer()
        asyncio.create_task(self.stayConnectedToPublisher())
        await self.startStreamService()

    async def startStreamService(self):
        await self.syncData()
        await self.makeSubscription()

    def returnPeerIp(self, peer: str = Union[str, None]) -> str:
        if peer is not None:
            return peer.split(':')[0]
        return self.publisherHost.split(':')[0]

    def returnPeerPort(self, peer: str = Union[str, None]) -> int:
        if peer is not None:
            return int(peer.split(':')[1])
        return int(self.publisherHost.split(':')[1])

    @property
    def isConnectedToPublisher(self):
        if hasattr(self, 'dataClient') and self.dataClient is not None and self.publisherHost is not None:
            return self.dataClient.isConnected(self.publisherHost)
        return False

    async def stayConnectedToPublisher(self):
        while True:
            await asyncio.sleep(10)
            if not self.isConnectedToPublisher:
                await self.connectToPeer()
                await self.startStreamService()


    async def connectToPeer(self) -> bool:
        ''' Connects to a peer to receive subscription if it has an active subscription to the stream '''
        async def _isPublisherActive(publisher: str) -> bool:
            ''' confirms if the publisher has the subscription stream in its available stream '''
            try:
                response = await self.dataClient.isStreamActive(
                            peerHost=self.returnPeerIp(publisher),
                            peerPort=self.returnPeerPort(publisher),
                            uuid=self.streamUuid)
                if response.status == DataServerApi.statusSuccess.value:
                    return True
                else:
                    raise Exception
            except Exception:
                warning('Failed to connect to an active Publisher ')
                return False

        while not self.isConnectedToPublisher:
            self.publisherHost = self.peerInfo.publishersIp[0]
            if await _isPublisherActive(self.publisherHost):
                return True
            self.peerInfo.subscribersIp = [
                ip for ip in self.peerInfo.subscribersIp if ip != self.dataClient.serverHostPort[0]
            ]
            self.rng.shuffle(self.peerInfo.subscribersIp)
            for subscriberIp in self.peerInfo.subscribersIp:
                if await _isPublisherActive(subscriberIp):
                    self.publisherHost = subscriberIp
                    return True
            self.publisherHost = None
            debug('Waiting for some time', print=True)
            # await asyncio.sleep(60*60)
            await asyncio.sleep(10)

    async def syncData(self):
        '''
        - this can be highly optimized. but for now we do the simple version
        - just ask for their entire dataset every time
            - if it's different than the df we got from our own dataserver,
              then tell dataserver to save this instead
            - replace what we have
        '''
        try:
            externalDataResponse = await self.dataClient.getRemoteStreamData(
                peerHost=self.returnPeerIp(), 
                peerPort=self.returnPeerPort(),
                uuid=self.streamUuid)
            if externalDataResponse.status == DataServerApi.statusSuccess.value:
                externalDf = externalDataResponse.data
                if not externalDf.equals(self.data) and len(externalDf) > 0: # TODO : maybe we can find a better logic so that we don't lose the host server's valuable data ( krishna )
                    response = await self.dataClient.insertStreamData(
                                    uuid=self.streamUuid,
                                    data=externalDf,
                                    replace=True
                                )
                    if response.status == DataServerApi.statusSuccess.value:
                        info("Data updated in server", color='green')
                    else:
                        raise Exception(externalDataResponse.senderMsg)
                else:
                    raise Exception(externalDataResponse.senderMsg)
        except Exception as e:
            error("Failed to sync data, ", e)

    # TODO: after subscribing let others know that we are subscribed to a particular data stream
    async def makeSubscription(self):
        '''
        - and subscribe to the stream so we get the information
            - whenever we get an observation on this stream, pass to the DataServer
        - continually generate predictions for prediction publication streams and pass that to
        '''
        await self.dataClient.subscribe(
              peerHost=self.returnPeerIp(),
              peerPort=self.returnPeerPort(),
              uuid=self.streamUuid,
              publicationUuid=self.predictionStreamUuid,
              callback=self.handleSubscriptionMessage)

    async def handleSubscriptionMessage(self, subscription: any,  message: Message, pubSubFlag: bool = False):
        if message.status == DataClientApi.streamInactive.value:
            self.publisherHost = None
            await self.connectToPeer()
            await self.startStreamService()
        else:
            await self.appendNewData(message.data, pubSubFlag)
            self.pauseAll()
            await self.producePrediction()
            self.resumeAll()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    async def appendNewData(self, observation: Union[pd.DataFrame, dict], pubSubFlag: bool):
        """extract the data and save it to self.data"""
        if pubSubFlag:
            parsedData = json.loads(observation.raw)
            if validate_single_entry(parsedData["time"], parsedData["data"]):
                await self.dataClient.insertStreamData(
                        uuid=self.streamUuid,
                        data=pd.DataFrame({ 'value': [float(parsedData["data"])]
                                    }, index=[str(parsedData["time"])]),
                        isSub=True
                    )
                self.data = pd.concat(
                    [
                        self.data,
                        pd.DataFrame({
                            "date_time": [str(parsedData["time"])],
                            "value": [float(parsedData["data"])],
                            "id": [str(parsedData["hash"])]})
                    ],
                    ignore_index=True)
            else:
                error("Row not added due to corrupt observation")
        else:
            observationDf = observation.reset_index().rename(columns={
                                'index': 'date_time',
                                'hash': 'id'
                            })
            if validate_single_entry(observationDf['date_time'].values[0], observationDf["value"].values[0]):
                self.data = pd.concat([self.data, observationDf], ignore_index=True)
            else:
                error("Row not added due to corrupt observation")

    async def passPredictionData(self, forecast: pd.DataFrame):
        try:
            response = await self.dataClient.insertStreamData(
                            uuid=self.predictionStreamUuid,
                            data=forecast,
                            isSub=True
                        )
            if response.status == DataServerApi.statusSuccess.value:
                info(response.senderMsg, color='green')
            else:
                raise Exception(response.senderMsg)
        except Exception as e:
            error('Failed to send Prediction to server : ', e)

    #pubsub functions


    async def producePrediction(self, updatedModel=None):
        """
        triggered by
            - model replaced with a better one
            - new observation on the stream
        """
        try:
            updatedModel = updatedModel or self.stable
            if updatedModel is not None:
                forecast = updatedModel.predict(data=self.data)
                if isinstance(forecast, pd.DataFrame):
                    predictionDf = pd.DataFrame({ 'value': [StreamForecast.firstPredictionOf(forecast)]
                                    }, index=[datetimeToTimestamp(now())])
                    await self.passPredictionData(predictionDf)
                else:
                    raise Exception('Forecast not in dataframe format')
        except Exception as e:
            error(e)
            await self.fallback_prediction()

    async def fallback_prediction(self):
        if os.path.isfile(self.modelPath()):
            try:
                os.remove(self.modelPath())
                debug("Deleted failed model file:", self.modelPath(), color="teal")
            except Exception as e:
                error(f'Failed to delete model file: {str(e)}')
        backupModel = self.defaultAdapters[-1]()
        try:
            trainingResult = backupModel.fit(data=self.data)
            if abs(trainingResult.status) == 1:
                await self.producePrediction(backupModel)
        except Exception as e:
            error(f"Error training new model: {str(e)}")

    async def loadData(self) -> pd.DataFrame:
        try:
            response = await self.dataClient.getLocalStreamData(uuid=self.streamUuid)
            if response.status == DataServerApi.statusSuccess.value:
                conformedData = response.data.reset_index().rename(columns={
                    'ts': 'date_time',
                    'hash': 'id'
                })
                del conformedData['provider']
                return conformedData
            else:
                raise Exception(response.senderMsg)
        except Exception as e:
            debug(e)
            return pd.DataFrame(columns=["date_time", "value", "id"])

    def modelPath(self) -> str:
        return (
            '/Satori/Neuron/models/veda/'
            f'{self.predictionStreamUuid}/'
            f'{self.adapter.__name__}.joblib')

    def chooseAdapter(self, inplace: bool = False) -> ModelAdapter:
        """
        everything can try to handle some cases
        Engine
            - low resources available - SKAdapter
            - few observations - SKAdapter
            - (mapping of cases to suitable adapters)
        examples: StartPipeline, SKAdapter, XGBoostPipeline, ChronosPipeline, DNNPipeline
        """
        # TODO: this needs to be aultered. I think the logic is not right. we
        #       should gather a list of adapters that can be used in the
        #       current condition we're in. if we're already using one in that
        #       list, we should continue using it until it starts to make bad
        #       predictions. if not, we should then choose the best one from the
        #       list - we should optimize after we gather acceptable options.

        if False: # for testing specific adapters
            adapter = XgbChronosAdapter
        else:
            import psutil
            availableRamGigs = psutil.virtual_memory().available / 1e9
            adapter = None
            for p in self.preferredAdapters:
                if p in self.failedAdapters:
                    continue
                if p.condition(data=self.data, cpu=self.cpu, availableRamGigs=availableRamGigs) == 1:
                    adapter = p
                    break
            if adapter is None:
                for adapter in self.defaultAdapters:
                    if adapter not in self.failedAdapters:
                        break
                if adapter is None:
                    adapter = self.defaultAdapters[-1]
        if (
            inplace and (
                not hasattr(self, 'pilot') or
                not isinstance(self.pilot, adapter))
        ):
            info(
                f'AI Engine: stream id {self.streamUuid} '
                f'switching from {self.adapter.__name__} '
                f'to {adapter.__name__} on {self.streamUuid}',
                color='teal')
            self.adapter = adapter
            self.pilot = adapter(uid=self.streamUuid)
            self.pilot.load(self.modelPath())
        return adapter

    async def run(self):
        """
        Async main loop for generating models and comparing them to the best known
        model so far in order to replace it if the new model is better, always
        using the best known model to make predictions on demand.
        """
        # for testing
        #if self.modelPath() != "/Satori/Neuron/models/veda/YyBHl6bN1GejAEyjKwEDmywFU-M-/XgbChronosAdapter.joblib":
        #    return
        while len(self.data) > 0:
            if self.paused:
                await asyncio.sleep(10)
                continue
            self.chooseAdapter(inplace=True)
            try:
                trainingResult = self.pilot.fit(data=self.data, stable=self.stable)
                if trainingResult.status == 1:
                    if self.pilot.compare(self.stable):
                        if self.pilot.save(self.modelPath()):
                            self.stable = copy.deepcopy(self.pilot)
                            info(
                                "stable model updated for stream:",
                                self.streamUuid,
                                print=True)
                            await self.producePrediction(self.stable)
                else:
                    debug(f'model training failed on {self.streamUuid} waiting 10 minutes to retry', print=True)
                    self.failedAdapters.append(self.pilot)
                    await asyncio.sleep(60)
            except Exception as e:
                import traceback
                traceback.print_exc()
                error(e)
                try:
                    debug(self.pilot.dataset)
                except Exception as e:
                    pass


    def run_forever(self):
        '''Creates separate threads for running the peer connections and model training loop'''

        def training_loop_thread():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.run())
                loop.close()
            except Exception as e:
                error(f"Error in training loop thread: {e}")
                import traceback
                traceback.print_exc()

        if self.transferProtocol == 'p2p' and self.transferProtocol == 'p2p-proactive':
            init_task = asyncio.create_task(self.p2pInit())

        self.thread = threading.Thread(target=training_loop_thread, daemon=True)
        self.thread.start()
