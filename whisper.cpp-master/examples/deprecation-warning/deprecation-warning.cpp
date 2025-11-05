// Warns users that this filename was deprecated, and provides a link for more information.

#include <cstdio>
#include <string>

// Main
int main(int argc, char** argv) {
    std::string filename = "main";
    if (argc >= 1) {
        filename = argv[0];
    }

    // Get only the program name from the full path
    size_t pos = filename.find_last_of("/\\");
    if (pos != std::string::npos) {
        filename = filename.substr(pos+1);
    }

    // Append "whisper-" to the beginning of filename to get the replacemnt filename
    std::string replacement_filename = "whisper-" + filename;

    // The exception is if the filename is "main", then our replacement filename is "whisper-cli"
    if (filename == "main") {
        replacement_filename = "whisper-cli";
    }

    if (filename == "main.exe") {
        replacement_filename = "whisper-cli.exe";
    }

    fprintf(stdout, "\n");
    fprintf(stdout, "WARNING: The binary '%s' is deprecated.\n", filename.c_str());
    fprintf(stdout, " Please use '%s' instead.\n", replacement_filename.c_str());
    fprintf(stdout, " See https://github.com/ggerganov/whisper.cpp/tree/master/examples/deprecation-warning/README.md for more information.\n");
    fprintf(stdout, "\n");

    return EXIT_FAILURE;
}
