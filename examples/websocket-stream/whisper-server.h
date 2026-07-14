#ifndef WHISPER_SERVER_H
#define WHISPER_SERVER_H
#include "server-params.h"
#include "ixwebsocket/IXWebSocketServer.h"
class WhisperServer {

    ix::WebSocketServer server;
public:
    WhisperServer(const ServerParams& params);
    ~WhisperServer();
    void run();
};
#endif
