#ifndef MESSAGE_BUFFER_H
#define MESSAGE_BUFFER_H
class MessageBuffer {
public:
    std::string connection_id;
    void add_message(const char* msg);

    std::string get_payload();

    void flush();

    void send_via_http(const std::string& data);
};
#endif
