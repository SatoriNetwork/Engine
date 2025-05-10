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
from satorilib.concepts import Observation, Stream
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
                            # warning(
                            #     'received:',
                            #     f'\n {obs.streamId.cleanId}',
                            #     f'\n ({obs.value}, {obs.observationTime}, {obs.observationHash})',
                            #     print=True)
                            streamModel = self.streamModels.get(obs.streamId.uuid)
                            if isinstance(streamModel, StreamModel) and getattr(streamModel, 'UsePubsub', True):
                                info(
                                    'received:',
                                    f'\n {obs.streamId.cleanId}',
                                    f'\n ({obs.value}, {obs.observationTime}, {obs.observationHash})',
                                    print=True)
                                # try:
                                #     task = asyncio.create_task(
                                #         streamModel.handleSubscriptionMessage(
                                #             "Subscription",
                                #             message=Message({
                                #                 'data': obs,
                                #                 'status': 'stream/observation'}),
                                #             pubSubFlag=True)
                                #     )
                                    
                                #     # Store the task reference to prevent garbage collection
                                #     self.threads.append(task)  # Rename self.threads to self.tasks for clarity
                                    
                                #     # Set up a callback to handle task completion and cleanup
                                #     def task_done_callback(completed_task):
                                #         try:
                                #             # Check for exceptions
                                #             if completed_task.exception():
                                #                 error(f"Exception in task: {completed_task.exception()}")
                                #             # Remove task from the list once completed
                                #             if completed_task in self.threads:
                                #                 self.threads.remove(completed_task)
                                #         except asyncio.CancelledError:
                                #             # Handle cancellation
                                #             pass
                                    
                                #     task.add_done_callback(task_done_callback)
                                # except Exception as e:
                                #     error(f"Failed to create task: {e}")

                                def run_async_in_thread():
                                    try:
                                        loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(loop)
                                        loop.run_until_complete(
                                            streamModel.handleSubscriptionMessage(
                                                "Subscription",
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
                                        # Remove this thread from the list of active threads
                                        if thread in self.threads:
                                            self.threads.remove(thread)
                                
                                thread = threading.Thread(target=run_async_in_thread)
                                thread.daemon = True
                                self.threads.append(thread)
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
                onDisconnect=onDisconnect)
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
                for _, streamModel in self.streamModels.items():
                    if hasattr(streamModel, 'dataClientOfIntServer'):
                        streamModel.updateDataClient(self.dataClient)
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
                transferProtocolPayload = pubSubResponse.streamInfo.get('transferProtocolPayload')
                transferProtocolKey = pubSubResponse.streamInfo.get('transferProtocolKey')
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
                if self.transferProtocol == 'p2p-pubsub' or self.transferProtocol == 'p2p-proactive-pubsub':
                    self.subConnect(key=transferProtocolKey)
                    return
            except Exception:
                warning(f"Failed to fetch pub-sub info, waiting for {waitingPeriod} seconds")
                await asyncio.sleep(waitingPeriod)

    async def stayConnectedForever(self):
        ''' alternative to await asyncio.Event().wait() '''
        while True:
            await asyncio.sleep(5)
            self.cleanupThreads()
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
                    identity=self.identity,
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
        identity: EvrmoreIdentity,
        pauseAll:callable,
        resumeAll:callable,
        transferProtocol: str
    ):
        streamModel = cls(
            streamUuid,
            predictionStreamUuid,
            peerInfo,
            dataClient,
            identity,
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
        identity: EvrmoreIdentity,
        pauseAll:callable,
        resumeAll:callable,
        transferProtocol: str
    ):
        self.cpu = getProcessorCount()
        self.pauseAll = pauseAll
        self.resumeAll = resumeAll
        # self.preferredAdapters: list[ModelAdapter] = [XgbChronosAdapter, XgbAdapter, StarterAdapter ]# SKAdapter #model[0] issue
        self.preferredAdapters: list[ModelAdapter] = [ XgbAdapter, StarterAdapter ]# SKAdapter #model[0] issue
        self.defaultAdapters: list[ModelAdapter] = [XgbAdapter, XgbAdapter, StarterAdapter]
        self.failedAdapters = []
        self.thread: threading.Thread = None
        self.streamUuid: str = streamUuid
        self.predictionStreamUuid: str = predictionStreamUuid
        self.peerInfo: PeerInfo = peerInfo
        self.dataClientOfIntServer: DataClient = dataClient
        self.identity: EvrmoreIdentity = identity
        self.rng = np.random.default_rng(37)
        self.publisherHost = None
        self.transferProtocol: str = transferProtocol
        self.usePubSub: bool = False
        self.currentPredictionTask = None
        # self.syncedPublishers = set() # variable used for not syncing everytime

    async def initialize(self):
        self.data: pd.DataFrame = await self.loadData()
        self.adapter: ModelAdapter = self.chooseAdapter()
        self.pilot: ModelAdapter = self.adapter(uid=self.streamUuid)
        self.pilot.load(self.modelPath())
        self.stable: ModelAdapter = copy.deepcopy(self.pilot)
        self.paused: bool = False
        self.dataClientOfExtServer: Union[DataClient, None] = DataClient(self.dataClientOfIntServer.serverHostPort[0], self.dataClientOfIntServer.serverPort, identity=self.identity)
        debug(f'AI Engine: stream id {self.streamUuid} using {self.adapter.__name__}', color='teal')

    # async def tempDataClientWithServer(self):
    #     ''' we instantiate a data client which has isLocal='engine' as Auth everytime we try to communicate with our own data stream and close it after the usage '''
    #     async def authenticate(dataClient: DataClient) -> bool:
    #         response = await dataClient.authenticate(islocal='engine')
    #         if response.status == DataServerApi.statusSuccess.value:
    #             info("Local Engine successfully connected to Server Ip at :", self.dataServerIp, color="green")
    #             return True
    #         return False

    #     async def initiateServerConnection() -> bool:
    #         ''' local engine client authorization '''
    #         dataClientTemp = DataClient(self.dataClientOfIntServer.serverHostPort[0], self.dataClientOfIntServer.serverPort, identity=self.identity)
    #         return await authenticate(dataClientTemp)

    #     while not self.isConnectedToServer:
    #         try:
    #             if await initiateServerConnection():
    #                 return True
    #         except Exception as e:
    #             error(f'Failed to connect to server', e)

    # Testing Purpose ( Don't Delete ): Add heartbeat mechanism to maintain connections during long processing
    async def maintain_connection(self):
        """Send periodic heartbeats to server to keep connection alive"""
        while True:
            if self.dataClientOfIntServer.isConnected():
                try:
                    # A lightweight ping-like request
                    await self.dataClientOfIntServer.isLocalEngineClient()
                except Exception:
                    pass
            await asyncio.sleep(15)  

    async def p2pInit(self):
        # await self.makeSubscription(self.dataClientOfIntServer.serverHostPort[0], True)
        # asyncio.create_task(self.maintain_connection()) # Testing
        await self.connectToPeer()
        asyncio.create_task(self.stayConnectedToPublisher())
        asyncio.create_task(self.checkIfPublisherStreamIsActive())
        await self.startStreamService()

    async def startStreamService(self):
        # The below helps to not sync everytime we connect to publisher, if any problems arises with sync then we can enable the below
        # if self.publisherHost not in self.syncedPublishers:
        #     await self.syncData()
        #     self.syncedPublishers.add(self.publisherHost)
        await self.syncData()
        await self.makeSubscription()

    def updateDataClient(self, dataClient):
        ''' Update the internal server data client reference '''
        self.dataClientOfIntServer = dataClient

    def returnPeerIp(self, peer: Union[str, None] = None) -> str:
        if peer is not None:
            return peer.split(':')[0]
        return self.publisherHost.split(':')[0]

    def returnPeerPort(self, peer: Union[str, None] = None) -> int:
        if peer is not None:
            return int(peer.split(':')[1])
        return int(self.publisherHost.split(':')[1])

    @property
    def isConnectedToPublisher(self):
        if hasattr(self, 'dataClientOfExtServer') and self.dataClientOfExtServer is not None and self.publisherHost is not None:
            return self.dataClientOfExtServer.isConnected(self.returnPeerIp(), self.returnPeerPort())
        return False

    @property
    def isConnectedToPublisher(self):
        if hasattr(self, 'dataClientOfExtServer') and self.dataClientOfExtServer is not None and self.publisherHost is not None:
            return self.dataClientOfExtServer.isConnected(self.returnPeerIp(), self.returnPeerPort())
        return False

    async def stayConnectedToPublisher(self):
        while True:
            await asyncio.sleep(9)
            if not self.isConnectedToPublisher:
                self.publisherHost = None
                await self.dataClientOfIntServer.streamInactive(self.streamUuid)
                await self.connectToPeer()
                await self.startStreamService()

    async def checkIfPublisherStreamIsActive(self):
        while True:
            await asyncio.sleep(60*5)
            if self.publisherHost is not None:
                if not self._isPublisherActive():
                    await self.dataClientOfIntServer.streamInactive(self.streamUuid)
                    await self.connectToPeer()
                    await self.startStreamService()
                
    async def _isPublisherActive(self, publisher: str = None) -> bool:
        ''' confirms if the publisher has the subscription stream in its available stream '''
        try:
            response = await self.dataClientOfExtServer.isStreamActive(
                        peerHost=self.returnPeerIp(publisher) if publisher else self.returnPeerIp(),
                        peerPort=self.returnPeerPort(publisher) if publisher else self.returnPeerIp(),
                        uuid=self.streamUuid)
            if response.status == DataServerApi.statusSuccess.value:
                return True
            else:
                raise Exception
        except Exception:
            # warning('Failed to connect to an active Publisher ', publisher)
            return False


    async def connectToPeer(self) -> bool:
        ''' Connects to a peer to receive subscription if it has an active subscription to the stream '''
        async def try_connection(ip):
            try:
                if await self._isPublisherActive(ip):
                    return (ip, True)
                return (ip, False)
            except Exception as e:
                return (ip, False)

        while not self.isConnectedToPublisher:
            if self.peerInfo.publishersIp is not None and len(self.peerInfo.publishersIp) > 0:
                self.publisherHost = self.peerInfo.publishersIp[0]
                if await self._isPublisherActive(self.publisherHost):
                    self.usePubSub = False
                    return True
            subscriber_ips = [ip for ip in self.peerInfo.subscribersIp]
            self.rng.shuffle(subscriber_ips)
            tasks = []
            for ip in subscriber_ips:
                task = asyncio.create_task(try_connection(ip))
                tasks.append(task)

            for future in asyncio.as_completed(tasks):
                try:
                    ip, is_active = await future
                    if is_active:
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        self.publisherHost = ip
                        self.usePubSub = False
                        return True
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    error(f"Error checking peer: {str(e)}")
            self.publisherHost = None
            warning('Failed to connect to Peers, switching to PubSub', self.streamUuid, print=True)
            self.usePubSub = True
            await asyncio.sleep(60*60)

    async def syncData(self):
        '''
        - this can be highly optimized. but for now we do the simple version
        - just ask for their entire dataset every time
            - if it's different than the df we got from our own dataserver,
              then tell dataserver to save this instead
            - replace what we have
        '''
        try:
            externalDataResponse = await self.dataClientOfExtServer.getRemoteStreamData(
                peerHost=self.returnPeerIp(),
                peerPort=self.returnPeerPort(),
                uuid=self.streamUuid)
            if externalDataResponse.status == DataServerApi.statusSuccess.value:
                externalDf = externalDataResponse.data
                if not externalDf.equals(self.data) and len(externalDf) > 0: # TODO : maybe we can find a better logic so that we don't lose the host server's valuable data ( krishna )
                    response = await self.dataClientOfIntServer.insertStreamData(
                                    uuid=self.streamUuid,
                                    data=externalDf,
                                    replace=True
                                )
                    if response.status == DataServerApi.statusSuccess.value:
                        info("Data updated in server", color='green')
                        externalDf = externalDf.reset_index().rename(columns={
                                    'ts': 'date_time',
                                    'hash': 'id'
                                }).drop(columns=['provider'])
                        self.data = externalDf
                    else:
                        raise Exception(externalDataResponse.senderMsg)
                else:
                    raise Exception(externalDataResponse.senderMsg)
        except Exception as e:
            error("Failed to sync data, ", e)

    # TODO: after subscribing let others know that we are subscribed to a particular data stream
    async def makeSubscription(self, peerHost: Union[str, None] = None, serverSubscribe: bool = False):
        '''
        - and subscribe to the stream so we get the information
            - whenever we get an observation on this stream, pass to the DataServer
        - continually generate predictions for prediction publication streams and pass that to
        '''
        if serverSubscribe:
            await self.dataClientOfIntServer.subscribe(
                peerHost=peerHost or self.returnPeerIp(),
                **(dict() if serverSubscribe is True else {'peerPort': self.returnPeerPort()}),
                uuid=self.streamUuid,
                publicationUuid=self.predictionStreamUuid,
                callback=self.handleSubscriptionMessage)
        else:
            await self.dataClientOfExtServer.subscribe(
                peerHost=peerHost or self.returnPeerIp(),
                **(dict() if serverSubscribe is True else {'peerPort': self.returnPeerPort()}),
                uuid=self.streamUuid,
                publicationUuid=self.predictionStreamUuid,
                callback=self.handleSubscriptionMessage)

    async def handleSubscriptionMessage(self, subscription: any,  message: Message, pubSubFlag: bool = False):
        if message.status == DataClientApi.streamInactive.value:
            print("Inactive")
            self.publisherHost = None
            await self.connectToPeer()
            await self.startStreamService()
        else:
            await self.appendNewData(message.data, pubSubFlag)
            self.pauseAll(force=True)
            # try:
            #     if self.currentPredictionTask is not None and not self.currentPredictionTask.done():
            #         self.currentPredictionTask.cancel()
            #         try:
            #             await asyncio.wait_for(asyncio.shield(self.currentPredictionTask), timeout=0.5)
            #         except (asyncio.CancelledError, asyncio.TimeoutError):
            #             pass
            #         error(f"Canceled existing prediction task for stream {self.streamUuid}")
            # except Exception as e:
            #     error(f"Error canceling task: {str(e)}")

            # self.currentPredictionTask = asyncio.create_task(self.producePrediction())
            await self.producePrediction()
            running_tasks = len([t for t in asyncio.all_tasks() if not t.done()])
            all_tasks = len([t for t in asyncio.all_tasks()])
            print("all_tasks", all_tasks)
            print(f"Total running tasks: {running_tasks}")
            self.resumeAll(force=True)

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    async def appendNewData(self, observation: Union[pd.DataFrame, dict], pubSubFlag: bool):
        """extract the data and save it to self.data"""
        print(observation)
        try:
            if pubSubFlag:
                parsedData = json.loads(observation.raw)
                if validate_single_entry(parsedData["time"], parsedData["data"]):
                        await self.dataClientOfIntServer.insertStreamData(
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
                observation_id = observation['hash'].values[0]
                # Check if self.data is not empty and if the ID already exists
                if not self.data.empty and observation_id in self.data['id'].values:
                    error("Row not added because observation with same ID already exists")
                elif validate_single_entry(observation.index[0], observation["value"].values[0]):
                    response = await self.dataClientOfIntServer.insertStreamData(
                            uuid=self.streamUuid,
                            data=observation,
                            isSub=True
                        )
                    if response.status == DataServerApi.statusSuccess.value:
                        info(response.senderMsg, color='green')
                    else:
                        raise Exception("Raw ", response.senderMsg)
                    observationDf = observation.reset_index().rename(columns={
                                'index': 'date_time',
                                'hash': 'id'
                            }).drop(columns=['provider'])
                    self.data = pd.concat([self.data, observationDf], ignore_index=True)
                else:
                    error("Row not added due to corrupt observation")
        except Exception as e:
            error("Subscription data not added", e)

    async def passPredictionData(self, forecast: pd.DataFrame):
        try:
            response = await self.dataClientOfIntServer.insertStreamData(
                            uuid=self.predictionStreamUuid,
                            data=forecast,
                            isSub=True
                        )
            if response.status == DataServerApi.statusSuccess.value:
                info("Prediction", response.senderMsg, color='green')
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
                loop = asyncio.get_event_loop()
                forecast = await loop.run_in_executor(
                    None,  # Uses default executor (ThreadPoolExecutor)
                    lambda: updatedModel.predict(data=self.data)
                )
                if isinstance(forecast, pd.DataFrame):
                    predictionDf = pd.DataFrame({ 'value': [StreamForecast.firstPredictionOf(forecast)]
                                    }, index=[datetimeToTimestamp(now())])
                    print(predictionDf)
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
            response = await self.dataClientOfIntServer.getLocalStreamData(uuid=self.streamUuid)
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
            availableSwapGigs = psutil.swap_memory().free / 1e9
            totalAvailableRamGigs = availableRamGigs + availableSwapGigs
            adapter = None
            for p in self.preferredAdapters:
                if p in self.failedAdapters:
                    continue
                if p.condition(data=self.data, cpu=self.cpu, availableRamGigs=totalAvailableRamGigs) == 1:
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
                    await asyncio.sleep(600)
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

        if hasattr(self, 'thread') and self.thread and self.thread.is_alive():
            warning(f"Thread for model {self.streamUuid} already running. Not creating another.")
            return

        # def training_loop_thread():
        #     try:
        #         loop = asyncio.new_event_loop()
        #         asyncio.set_event_loop(loop)
        #         loop.run_until_complete(self.run())
        #         loop.close()
        #     except Exception as e:
        #         error(f"Error in training loop thread: {e}")
        #         import traceback
        #         traceback.print_exc()

        def training_loop_thread():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.run())
                finally:
                    loop.close()
            except Exception as e:
                error(f"Error in training loop thread: {e}")
                import traceback
                traceback.print_exc()

        if self.transferProtocol == 'p2p-pubsub' or self.transferProtocol == 'p2p-proactive-pubsub':
            init_task = asyncio.create_task(self.p2pInit())

        self.thread = threading.Thread(target=training_loop_thread, daemon=True)
        self.thread.start()
