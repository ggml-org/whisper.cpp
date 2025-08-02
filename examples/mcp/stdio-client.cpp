#include "stdio-client.hpp"
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>

namespace mcp {

StdioClient::StdioClient() : server_pid(-1), server_stdin(nullptr), server_stdout(nullptr),
    server_stderr(nullptr) ,request_id_counter(0) , server_running(false) {
    stdin_pipe[0]  = stdin_pipe[1]  = -1;
    stdout_pipe[0] = stdout_pipe[1] = -1;
    stderr_pipe[0] = stderr_pipe[1] = -1;
}

StdioClient::~StdioClient() {
    cleanup();
}

void StdioClient::cleanup() {
    if (server_stdin) {
        fclose(server_stdin);
        server_stdin = nullptr;
    }

    if (server_stdout) {
        fclose(server_stdout);
        server_stdout = nullptr;
    }

    if (server_stderr) {
        fclose(server_stderr);
        server_stderr = nullptr;
    }

    if (server_running && server_pid > 0) {
        kill(server_pid, SIGTERM);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        int status;
        if (waitpid(server_pid, &status, WNOHANG) == 0) {
            kill(server_pid, SIGKILL);
            waitpid(server_pid, &status, 0);
        }
        server_running = false;
    }
}

bool StdioClient::start_server(const std::string & server_command, const std::vector<std::string> & args) {
    if (server_running) {
        return false; // Already running
    }

    // Create pipes
    if (pipe(stdin_pipe) == -1 || pipe(stdout_pipe) == -1 || pipe(stderr_pipe) == -1) {
        return false;
    }

    server_pid = fork();
    if (server_pid == -1) {
        return false;
    }

    if (server_pid == 0) {
        // Child process - become the server
        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);

        // Close all pipe ends
        close(stdin_pipe[0]); close(stdin_pipe[1]);
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        close(stderr_pipe[0]); close(stderr_pipe[1]);

        // Prepare arguments for execvp
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(server_command.c_str()));

        for (const auto& arg : args) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        argv.push_back(nullptr);

        execvp(server_command.c_str(), argv.data());
        exit(1); // exec failed
    }

    // Parent process - set up communication
    close(stdin_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    server_stdin = fdopen(stdin_pipe[1], "w");
    server_stdout = fdopen(stdout_pipe[0], "r");
    server_stderr = fdopen(stderr_pipe[0], "r");

    if (!server_stdin || !server_stdout || !server_stderr) {
        cleanup();
        return false;
    }

    server_running = true;
    return true;
}

void StdioClient::stop_server() {
    cleanup();
}

json StdioClient::send_request(const json & request) {
    if (!server_running) {
        throw std::runtime_error("Server is not running");
    }

    std::string request_str = request.dump() + "\n";

    if (fputs(request_str.c_str(), server_stdin) == EOF) {
        throw std::runtime_error("Failed to send request to server");
    }
    fflush(server_stdin);

    // For notifications (no id), don't wait for response
    if (!request.contains("id")) {
        return json{};
    }

    // Read response
    char buffer[4096];
    if (fgets(buffer, sizeof(buffer), server_stdout) == nullptr) {
        throw std::runtime_error("Failed to read response from server");
    }

    std::string response_str(buffer);
    if (!response_str.empty() && response_str.back() == '\n') {
        response_str.pop_back();
    }

    return json::parse(response_str);
}

void StdioClient::read_server_logs() {
    int flags = fcntl(fileno(server_stderr), F_GETFL, 0);
    fcntl(fileno(server_stderr), F_SETFL, flags | O_NONBLOCK);

    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), server_stderr) != nullptr) {
        std::cout << "[SERVER LOG] " << buffer;
    }

    fcntl(fileno(server_stderr), F_SETFL, flags);
}

json StdioClient::initialize(const std::string & client_name, const std::string & client_version) {
    json request = {
        {"jsonrpc", "2.0"},
        {"id", next_request_id()},
        {"method", "initialize"},
        {"params", {
            {"protocolVersion", "2024-11-05"},
            {"capabilities", {
                {"tools", json::object()}
            }},
            {"clientInfo", {
                {"name", client_name},
                {"version", client_version}
            }}
        }}
    };

    return send_request(request);
}

void StdioClient::send_initialized() {
    json notification = {
        {"jsonrpc", "2.0"},
        {"method", "notifications/initialized"}
    };

    send_request(notification);
}

json StdioClient::list_tools() {
    json request = {
        {"jsonrpc", "2.0"},
        {"id", next_request_id()},
        {"method", "tools/list"}
    };

    return send_request(request);
}

json StdioClient::call_tool(const std::string & tool_name, const json & arguments) {
    json request = {
        {"jsonrpc", "2.0"},
        {"id", next_request_id()},
        {"method", "tools/call"},
        {"params", {
            {"name", tool_name},
            {"arguments", arguments}
        }}
    };

    return send_request(request);
}

int StdioClient::next_request_id() {
    return ++request_id_counter;
}

bool StdioClient::wait_for_server_ready(int timeout_ms) {
    auto start = std::chrono::steady_clock::now();

    while (std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count() < timeout_ms) {

        if (server_running) {
            // Give server a moment to fully start up
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return true;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return false;
}

std::string StdioClient::get_last_server_logs() {
    std::stringstream logs;

    int flags = fcntl(fileno(server_stderr), F_GETFL, 0);
    fcntl(fileno(server_stderr), F_SETFL, flags | O_NONBLOCK);

    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), server_stderr) != nullptr) {
        logs << buffer;
    }

    fcntl(fileno(server_stderr), F_SETFL, flags);
    return logs.str();
}

} // namespace mcp
