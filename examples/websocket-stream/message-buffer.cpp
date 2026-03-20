#include <sstream>
#include <mutex>
#include <curl/curl.h>

#include "message-buffer.h"
extern std::string forward_url;
extern size_t max_messages;
namespace {
    std::stringstream ss;
    std::mutex mtx;
    size_t current_count = 0;
    static size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
        ((std::string*)userdata)->append(ptr, size * nmemb);
        return size * nmemb;
    }
}

void MessageBuffer::add_message(const char* msg) {
    std::lock_guard<std::mutex> lock(mtx);
    ss << std::string(msg) << '\n';
    if (++current_count >= max_messages) {
        flush();
    }
}

std::string MessageBuffer::get_payload() {
    std::lock_guard<std::mutex> lock(mtx);
    return ss.str();
}

void MessageBuffer::flush() {
    std::string payload = get_payload();
    if (!payload.empty()) {
        send_via_http(payload);
        ss.str("");  //clear string stream
        current_count = 0;
    }
}

void MessageBuffer::send_via_http(const std::string& data) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        printf("CURL init failed");
        return;
    }

    //make headers
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: text/plain");
    std::string cid_header = "X-Connection-ID: " + connection_id;
    headers = curl_slist_append(headers, cid_header.c_str());

    //config curl
    std::string response;
    printf("sending to %s\n", forward_url.c_str());
    curl_easy_setopt(curl, CURLOPT_URL, forward_url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, data.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 2L);

    //run curl
    for (int retry = 0; retry < 3; ++retry) {
        CURLcode res = curl_easy_perform(curl);
        if (res == CURLE_OK) {
             printf("[Response (%s): %s\n", connection_id.c_str(), response.c_str());
             break;
        }
        printf("[CURL error:  %s\n", curl_easy_strerror(res));
    }

    //clean
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
}
