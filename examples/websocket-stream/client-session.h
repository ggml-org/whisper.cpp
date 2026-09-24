#ifndef CLIENT_SESSION_H
#define CLIENT_SESSION_H
#include <vector>
#include <mutex>
#include <atomic>
#include "ixwebsocket/IXWebSocketServer.h"
#include "message-buffer.h"
struct ClientSession {
    std::vector<float> pcm_buffer;
    std::mutex mtx;
    std::atomic<bool> active{true};
    ix::WebSocket *connection;
    MessageBuffer buffToBackend;
};
#endif
