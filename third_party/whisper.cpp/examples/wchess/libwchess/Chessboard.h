#pragma once
#include <string>
#include <set>
#include <memory>

// just basic validation
// fixme: missing en passant, castling, promotion, etc.
struct State;
class Piece;
class Chessboard {
public:
    Chessboard();
    ~Chessboard();
    std::string process(const std::string& command);
    std::string stringifyBoard();
    const std::string& grammar() { return m_grammar; }
    const std::string& prompt() { return m_prompt; }
    void setPrompt(const std::string& prompt);
private:
    bool parseCommand(const std::string& command, Piece*& piece, char& pos_to);
    bool move(Piece& piece, char pos);
    void flagUpdates(char pos_from, char pos_to);
    void updatePins(Piece& piece);
    void detectChecks();
    void setGrammar();

    std::unique_ptr<State> m_state;
    std::set<char> m_allowedInCheck;
    bool m_inCheck = false;
    int m_moveCounter = 0;
    std::string m_grammar;
    std::string m_prompt;
};
