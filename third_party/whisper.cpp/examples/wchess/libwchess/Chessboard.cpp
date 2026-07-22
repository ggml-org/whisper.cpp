#include "Chessboard.h"

#include <array>
#include <vector>
#include <algorithm>
#include <cstring>
#include <set>
#include <list>
#include <chrono>

namespace {
constexpr std::array<const char*, 64> positions = {
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
};
constexpr char INVALID_POS = positions.size();
constexpr int R = 0; // rank index
constexpr int F = 1; // file index
#define FILE (c[F] - '1')
#define RANK (c[R] - 'a')
constexpr char operator ""_P(const char * c, size_t size) {
    return size < 2 || RANK < 0 || RANK > 7 ||
        FILE < 0 || FILE > 7 ? INVALID_POS : FILE * 8 + RANK;
}
#undef FILE
#undef RANK

struct sview {
    const char * ptr = nullptr;
    size_t size = 0;

    sview() = default;
    sview(const char * p, size_t s) : ptr(p), size(s) {}
    sview(const std::string& s) : ptr(s.data()), size(s.size()) {}

    size_t find(char del, size_t pos) {
        while (pos < size && ptr[pos] != del) ++pos;
        return pos < size ? pos : std::string::npos;
    }
};

std::vector<sview> split(sview str, char del) {
    std::vector<sview> res;
    size_t cur = 0;
    size_t last = 0;
    while (cur != std::string::npos) {
        if (str.ptr[last] == ' ') {
            ++last;
            continue;
        }
        cur = str.find(del, last);
        size_t len = cur == std::string::npos ? str.size - last : cur - last;
        res.emplace_back(str.ptr + last, len);
        last = cur + 1;
    }
    return res;
}

char strToPos(sview str) {
    return operator ""_P(str.ptr, str.size);
}

constexpr std::array<const char*, 6> pieceNames =  {
    "pawn", "knight", "bishop", "rook", "queen", "king",
};

static constexpr std::array<char, 6> blackShort =  {
    'p', 'n', 'b', 'r', 'q', 'k',
};
static constexpr std::array<char, 6> whiteShort =  {
    'P', 'N', 'B', 'R', 'Q', 'K',
};

char strToType(sview str) {
    auto it = std::find_if(pieceNames.begin(), pieceNames.end(), [str] (const char* name) { return strncmp(name, str.ptr, str.size) == 0; });
    return it != pieceNames.end() ? it - pieceNames.begin() : pieceNames.size();
}

// directions
using Direction = std::array<char, 2>;

constexpr Direction N   = {(char)  0, (char)  1};
constexpr Direction NNE = {(char)  1, (char)  2};
constexpr Direction NE  = {(char)  1, (char)  1};
constexpr Direction ENE = {(char)  2, (char)  1};
constexpr Direction E   = {(char)  1, (char)  0};
constexpr Direction ESE = {(char)  2, (char) -1};
constexpr Direction SE  = {(char)  1, (char) -1};
constexpr Direction SSE = {(char)  1, (char) -2};
constexpr Direction S   = {(char)  0, (char) -1};
constexpr Direction SSW = {(char) -1, (char) -2};
constexpr Direction SW  = {(char) -1, (char) -1};
constexpr Direction WSW = {(char) -2, (char) -1};
constexpr Direction W   = {(char) -1, (char)  0};
constexpr Direction WNW = {(char) -2, (char)  1};
constexpr Direction NW  = {(char) -1, (char)  1};
constexpr Direction NNW = {(char) -1, (char)  2};

char makeStep(char pos, const Direction& d) {
    char next[2] = { char(positions[pos][R] + d[R]) , char(positions[pos][F] + d[F]) };
    return strToPos(sview{next, sizeof(next)});
}

template<class Modifier>
char traverse(char pos, const Direction& d, const Modifier& m, int count = 8) {
    while (--count >= 0) {
        pos = makeStep(pos, d);
        if (pos == INVALID_POS || m(pos)) break;
    }
    return pos;
}

Direction normalize(const Direction& distance) {
    //return {char((distance[R] > 0) - (distance[R] < 0)), char((distance[F] > 0) - (distance[F] < 0))};
    const int drp = distance[R] > 0 ? 1 : 0;
    const int drn = distance[R] < 0 ? 1 : 0;
    const int dfp = distance[F] > 0 ? 1 : 0;
    const int dfn = distance[F] < 0 ? 1 : 0;
    return {char(drp - drn), char(dfp - dfn)};
}

struct Pin {
    Direction d;
    Piece* pinner;
    Piece* pinned;
};
using Pins = std::list<Pin>;
using Board = std::array<Piece*, 64>;

std::vector<Direction> filter(const Direction& pin, std::initializer_list<Direction> directions) {
    if (pin[R] == 0 && pin[F] == 0) return directions;
    std::vector<Direction> result;
    for (auto& d : directions) {
        if ((d[R] == pin[R] || d[R] == -pin[R]) && (d[F] == pin[F] || d[F] == -pin[F])) result.push_back(d);
    }
    return result;
}
}

class Piece {
public:
    enum Types : char {
        Pawn,
        Knight,
        Bishop,
        Rook,
        Queen,
        King,
        //
        NUM_PIECES
    };

    enum Colors : char {
        White,
        Black,
    };

    const char* name() const;
    char initial() const;
    Types type() const { return m_type; }
    Colors color() const { return m_color; }
    char pos() const { return m_pos; }
    void setPos(char pos) {
        m_pos = pos;
        invalidate();
    }
    const char* coord() const;
    const std::set<char>& allowed() const { return m_allowed; }
    bool canReach(char pos) const;
    virtual bool movePattern(char pos) const = 0;
    void take();
    virtual void reinit(const State& state) = 0;
    void invalidate();
protected:
    Piece(Types type, Colors color, char pos, std::set<char> allowed)
        : m_type(type), m_color(color), m_pos(pos), m_allowed(std::move(allowed)) {}
    Piece(const Piece&) = delete;
    ~Piece() = default;

    const Types m_type;
    const Colors m_color;
    char m_pos;
    std::set<char> m_allowed;
    bool m_update = false;
};

struct Pawn : public Piece {
    Pawn(Colors color, char pos, std::set<char> next) : Piece(Types::Pawn, color, pos, std::move(next)) {}

    bool is_first_move() const {
        return m_color ? coord()[F] == '7' : coord()[F] == '2';
    }

    virtual bool movePattern(char pos) const override {
        if (m_pos == INVALID_POS) return false;
        auto cur = coord();
        auto next = positions[pos];
        Direction distance = {char(next[R] - cur[R]), char(next[F] - cur[F])};
        char forward = m_color ? -1 : 1;
        return (forward == distance[F] && distance[R] * distance[R] <= 1)
            || (is_first_move() && 2 * forward == distance[F] && distance[R] == 0);
    }

    virtual void reinit(const State& state) override;
};

struct Knight : public Piece {
    Knight(Colors color, char pos, std::set<char> next) : Piece(Types::Knight, color, pos, std::move(next)) {}

    virtual bool movePattern(char pos) const override {
        if (m_pos == INVALID_POS) return false;
        auto cur = coord();
        auto next = positions[pos];
        Direction diff = {char(next[R] - cur[R]), char(next[F] - cur[F])};
        return diff[R]*diff[R] + diff[F]*diff[F] == 5;
    }

    virtual void reinit(const State& state) override;
};

struct Bishop : public Piece {
    Bishop(Colors color, char pos) : Piece(Types::Bishop, color, pos, {}) {}

    virtual bool movePattern(char pos) const override {
        if (m_pos == INVALID_POS) return false;
        auto cur = coord();
        auto next = positions[pos];
        return cur[R] - cur[F] == next[R] - next[F] || cur[R] + cur[F] == next[R] + next[F];
    }

    virtual void reinit(const State& state) override;
};

struct Rook : public Piece {
    Rook(Colors color, char pos) : Piece(Types::Rook, color, pos, {}) {}

    virtual bool movePattern(char pos) const override {
        if (m_pos == INVALID_POS) return false;
        auto cur = coord();
        auto next = positions[pos];
        return cur[R] == next[R] || cur[F] == next[F];
    }

    virtual void reinit(const State& state) override;
};

struct Queen : public Piece {
    Queen(Colors color, char pos) : Piece(Types::Queen, color, pos, {}) {}

    virtual bool movePattern(char pos) const override {
        if (m_pos == INVALID_POS) return false;
        auto cur = coord();
        auto next = positions[pos];
        return cur[R] == next[R] || cur[F] == next[F] || cur[R] - cur[F] == next[R] - next[F] || cur[R] + cur[F] == next[R] + next[F];
    }

    virtual void reinit(const State& state) override;
};

struct King : public Piece {
    King(Colors color, char pos) : Piece(Types::King, color, pos, {}) {}

    virtual bool movePattern(char pos) const override {
        if (m_pos == INVALID_POS) return false;
        auto cur = coord();
        auto next = positions[pos];
        Direction diff = {char(next[R] - cur[R]), char(next[F] - cur[F])};
        return diff[R]*diff[R] + diff[F]*diff[F] <= 2;
    }

    virtual void reinit(const State& state) override;
};

struct PieceSet {
    Piece* begin() { return &p1; }
    Piece* end() { return &r2 + 1; }
    const Piece* begin() const { return &p1; }
    const Piece* end() const { return &r2 + 1; }
    Piece& operator[](int i) { return *(begin() + i); }
    const Piece& operator[](int i) const { return *(begin() + i); }

    Pawn   p1;
    Pawn   p2;
    Pawn   p3;
    Pawn   p4;
    Pawn   p5;
    Pawn   p6;
    Pawn   p7;
    Pawn   p8;
    Rook   r1;
    Knight n1;
    Bishop b1;
    Queen  q;
    King   k;
    Bishop b2;
    Knight n2;
    Rook   r2;
};

struct State {
    State();
    PieceSet blacks;
    PieceSet whites;
    Board board;
    Pins blackPins;
    Pins whitePins;
};

Direction findPin(const Piece& piece, const State& state) {
    auto& pins = piece.color() ? state.blackPins : state.whitePins;
    auto it = std::find_if(pins.begin(), pins.end(), [&] (const Pin& pin) { return pin.pinned == &piece; });
    if (it != pins.end()) return it->d;
    return {0, 0};
}

struct Find {
    Find(const Board& board) : m_board(board) {}
    bool operator() (char pos) const { return m_board[pos]; }
    const Board& m_board;
};

struct Add {
    Add(const Board& board, std::set<char>& moves, Piece::Colors color) : m_board(board), m_moves(moves), m_color(color) {}
    bool operator() (char pos) const {
        if (!m_board[pos] || m_board[pos]->color() != m_color) m_moves.insert(pos);
        return m_board[pos];
    }
    const Board& m_board;
    std::set<char>& m_moves;
    Piece::Colors m_color;
};

void Pawn::reinit(const State& state) {
    if (m_pos == INVALID_POS) return;
    if (!m_update) return;
    m_update = false;
    m_allowed.clear();

    auto pin = findPin(*this, state);

    auto & left = m_color ? SW : NW;
    auto & right = m_color ? SE : NE;

    for (auto& direction : filter(pin, { left, right })) {
        auto pos = makeStep(m_pos, direction);
        if (pos != INVALID_POS && state.board[pos] && state.board[pos]->color() != m_color) m_allowed.insert(pos);
    }

    auto & forward = m_color ? S : N;
    if (!filter(pin, {forward}).empty()) {
        traverse(m_pos, forward, [&] (char pos) {
                if (!state.board[pos]) m_allowed.insert(pos);
                return state.board[pos] || !is_first_move();
            }, 2);
    }
}

void Knight::reinit(const State& state) {
    if (m_pos == INVALID_POS) return;
    if (!m_update) return;
    m_update = false;
    m_allowed.clear();
    auto pin = findPin(*this, state);
    if (pin[R] != 0 || pin[F] != 0) return;
    for (auto& direction : { NNE, ENE, ESE, SSE, SSW, WSW, WNW, NNW }) {
        auto pos = makeStep(m_pos, direction);
        if (pos != INVALID_POS && (!state.board[pos] || state.board[pos]->color() != m_color)) m_allowed.insert(pos);
    }
}

void Bishop::reinit(const State& state) {
    if (m_pos == INVALID_POS) return;
    if (!m_update) return;
    m_update = false;
    m_allowed.clear();
    auto pin = findPin(*this, state);
    for (auto& direction : filter(pin, { NE, SE, SW, NW })) {
        traverse(m_pos, direction, Add(state.board, m_allowed, m_color));
    }
}

void Rook::reinit(const State& state) {
    if (m_pos == INVALID_POS) return;
    if (!m_update) return;
    m_update = false;
    m_allowed.clear();
    auto pin = findPin(*this, state);
    for (auto& direction : filter(pin, { N, E, S, W })) {
        traverse(m_pos, direction, Add(state.board, m_allowed, m_color));
    }
}

void Queen::reinit(const State& state) {
    if (m_pos == INVALID_POS) return;
    if (!m_update) return;
    m_update = false;
    m_allowed.clear();
    auto pin = findPin(*this, state);
    for (auto& direction : filter(pin, { N, NE, E, SE, S, SW, W, NW })) {
        traverse(m_pos, direction, Add(state.board, m_allowed, m_color));
    }
}

void King::reinit(const State& state) {
    if (m_pos == INVALID_POS) return;
    if (!m_update) return;
    m_update = false;
    m_allowed.clear();
    auto& enemyPieces = m_color ? state.whites : state.blacks;
    auto& pawnAttackLeft = m_color ? SW : NW;
    auto& pawnAttackRight = m_color ? SE : NE;
    for (auto& direction : { N, NE, E, SE, S, SW, W, NW }) {
        auto pos = makeStep(m_pos, direction);
        bool accept = pos != INVALID_POS && !(state.board[pos] && state.board[pos]->color() == m_color);
        if (accept) {
            for (auto& p : enemyPieces) {
                if (!p.movePattern(pos)) continue;
                if (p.type() == Piece::Knight || p.type() == Piece::King) {
                    accept = false;
                    break;
                }
                else if (p.type() == Piece::Pawn) {
                    auto from = positions[pos];
                    auto to = p.coord();
                    Direction d {char(to[R] - from[R]), char(to[F] - from[F])};
                    if (d == pawnAttackLeft || d == pawnAttackRight) {
                        accept = false;
                        break;
                    }
                }
                else {
                    auto from = positions[pos];
                    auto to = p.coord();
                    Direction d = normalize({char(to[R] - from[R]), char(to[F] - from[F])});
                    auto reached = traverse(pos, d, Find(state.board));
                    if (p.pos() == reached) {
                        accept = false;
                        break;
                    }
                }
            }
        }
        if (accept) m_allowed.insert(pos);
    }
}

const char* Piece::name() const {
    static_assert(pieceNames.size() == Piece::NUM_PIECES, "Mismatch between piece names and types");
    return pieceNames[m_type];
}

char Piece::initial() const {
    static_assert(blackShort.size() == Piece::NUM_PIECES, "Mismatch between piece names and types");
    static_assert(whiteShort.size() == Piece::NUM_PIECES, "Mismatch between piece names and types");
    return m_color ? blackShort[m_type] : whiteShort[m_type];
}

void Piece::invalidate() {
    m_update = true;
}


const char* Piece::coord() const {
    if (m_pos == INVALID_POS) return "";
    return positions[m_pos];
}

bool Piece::canReach(char pos) const {
    return movePattern(pos) && m_allowed.count(pos);
}

void Piece::take() {
    m_pos = INVALID_POS;
    m_allowed = {};
}

State::State()
    : blacks {
        {Piece::Black, "a7"_P, {"a5"_P, "a6"_P} },
        {Piece::Black, "b7"_P, {"b5"_P, "b6"_P} },
        {Piece::Black, "c7"_P, {"c5"_P, "c6"_P} },
        {Piece::Black, "d7"_P, {"d5"_P, "d6"_P} },
        {Piece::Black, "e7"_P, {"e5"_P, "e6"_P} },
        {Piece::Black, "f7"_P, {"f5"_P, "f6"_P} },
        {Piece::Black, "g7"_P, {"g5"_P, "g6"_P} },
        {Piece::Black, "h7"_P, {"h5"_P, "h6"_P} },
        {Piece::Black, "a8"_P},
        {Piece::Black, "b8"_P, {"a6"_P, "c6"_P} },
        {Piece::Black, "c8"_P},
        {Piece::Black, "d8"_P},
        {Piece::Black, "e8"_P},
        {Piece::Black, "f8"_P},
        {Piece::Black, "g8"_P, {"f6"_P, "h6"_P} },
        {Piece::Black, "h8"_P},
    }
    , whites {
        {Piece::White, "a2"_P, {"a3"_P, "a4"_P} },
        {Piece::White, "b2"_P, {"b3"_P, "b4"_P} },
        {Piece::White, "c2"_P, {"c3"_P, "c4"_P} },
        {Piece::White, "d2"_P, {"d3"_P, "d4"_P} },
        {Piece::White, "e2"_P, {"e3"_P, "e4"_P} },
        {Piece::White, "f2"_P, {"f3"_P, "f4"_P} },
        {Piece::White, "g2"_P, {"g3"_P, "g4"_P} },
        {Piece::White, "h2"_P, {"h3"_P, "h4"_P} },
        {Piece::White, "a1"_P},
        {Piece::White, "b1"_P, {"a3"_P, "c3"_P} },
        {Piece::White, "c1"_P},
        {Piece::White, "d1"_P},
        {Piece::White, "e1"_P},
        {Piece::White, "f1"_P},
        {Piece::White, "g1"_P, {"f3"_P, "h3"_P} },
        {Piece::White, "h1"_P},
    }
    , board {{
        &whites[ 8],  &whites[ 9],  &whites[10],  &whites[11],  &whites[12],  &whites[13],  &whites[14],  &whites[15],
        &whites[ 0],  &whites[ 1],  &whites[ 2],  &whites[ 3],  &whites[ 4],  &whites[ 5],  &whites[ 6],  &whites[ 7],
        nullptr,      nullptr,      nullptr,      nullptr,      nullptr,      nullptr,      nullptr,      nullptr,
        nullptr,      nullptr,      nullptr,      nullptr,      nullptr,      nullptr,      nullptr,      nullptr,
        nullptr,      nullptr,      nullptr,      nullptr,      nullptr,      nullptr,      nullptr,      nullptr,
        nullptr,      nullptr,      nullptr,      nullptr,      nullptr,      nullptr,      nullptr,      nullptr,
        &blacks[ 0],  &blacks[ 1],  &blacks[ 2],  &blacks[ 3],  &blacks[ 4],  &blacks[ 5],  &blacks[ 6],  &blacks[ 7],
        &blacks[ 8],  &blacks[ 9],  &blacks[10],  &blacks[11],  &blacks[12],  &blacks[13],  &blacks[14],  &blacks[15],
    }}
{}

Chessboard::Chessboard()
    : m_state(new State())
{
    setGrammar();
}

Chessboard::~Chessboard() = default;

void Chessboard::setPrompt(const std::string& prompt) {
    m_prompt = prompt;
    setGrammar();
}

void Chessboard::setGrammar() {
    m_grammar.clear();

    std::string result;
    if (m_prompt.empty()) {
        result += "move ::= \" \" ((piece | frompos) \" \" \"to \"?)? topos\n";
        //result += "move ::= \" \" frompos \" \" \"to \"? topos\n";
    }
    else {
        // result += "move ::= prompt \" \" ((piece | frompos) \" \" \"to \"?)? topos\n"
        result += "move ::= prompt \" \" frompos \" \" \"to \"? topos\n"
        "prompt ::= \" " + m_prompt + "\"\n";
    }

    std::set<Piece::Types> pieceTypes;
    std::set<char> from_pos;
    std::set<char> to_pos;
    auto& pieces =  m_moveCounter % 2 ? m_state->blacks : m_state->whites;
    std::set<size_t> flags;
    for (auto& p : pieces) {
        if (p.allowed().empty()) continue;
        bool addPiece = false;
        if (!m_inCheck || p.type() == Piece::King) {
            to_pos.insert(p.allowed().begin(), p.allowed().end());
            addPiece = !p.allowed().empty();
        }
        else {
            for (auto move : p.allowed()) {
                if (m_allowedInCheck.count(move)) {
                    to_pos.insert(move);
                    addPiece = true;
                }
            }
        }
        if (addPiece) {
            pieceTypes.insert(p.type());
            from_pos.insert(p.pos());
        }
    }
    if (pieceTypes.empty()) return;

    result += "piece ::= (";
    for (auto& p : pieceTypes) result += " \"" + std::string(pieceNames[p]) + "\" |";
    result.pop_back();
    result += ")\n\n";

    result += "frompos ::= (";
    for (auto& p : from_pos) result += " \"" + std::string(positions[p]) + "\" |";
    result.pop_back();
    result += ")\n";

    result += "topos ::= (";
    for (auto& p : to_pos) result += " \"" + std::string(positions[p]) + "\" |";
    result.pop_back();
    result += ")\n";

    m_grammar = std::move(result);
}

std::string Chessboard::stringifyBoard() {
    std::string result;
    result.reserve(16 + 2 * 64 + 16);
    for (char rank = 'a'; rank <= 'h'; ++rank) {
        result.push_back(rank);
        result.push_back(' ');
    }
    result.back() = '\n';
    for (int i = 7; i >= 0; --i) {
        for (int j = 0; j < 8; ++j) {
            auto p = m_state->board[i * 8 + j];
            if (p) result.push_back(p->initial());
            else result.push_back((i + j) % 2 ? '.' : '*');
            result.push_back(' ');
        }
        result.push_back('0' + i + 1);
        result.push_back('\n');
    }
    return result;
}

std::string Chessboard::process(const std::string& command) {
    const auto t_start = std::chrono::high_resolution_clock::now();
    auto color = Piece::Colors(m_moveCounter % 2);
    Piece* piece = nullptr;
    auto pos_to = INVALID_POS;
    if (!parseCommand(command, piece, pos_to)) return "";

    auto pos_from = piece->pos();

    if (!move(*piece, pos_to)) return "";

    flagUpdates(pos_from, pos_to);

    detectChecks();

    auto& enemyPieces = color ? m_state->whites : m_state->blacks;
    for (auto& p : enemyPieces) p.reinit(*m_state); // only enemy moves needed next

    std::string result = {positions[pos_from][R], positions[pos_from][F], '-', positions[pos_to][R], positions[pos_to][F]};
    ++m_moveCounter;
    setGrammar();
    const auto t_end = std::chrono::high_resolution_clock::now();
    auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    fprintf(stdout, "%s: Move '%s%s%s', (t = %d ms)\n", __func__, "\033[1m", result.data(), "\033[0m", (int) t_ms);
    if (m_grammar.empty()) result.push_back('#');
    return result;
}

bool Chessboard::parseCommand(const std::string& command, Piece*& piece, char& pos_to) {
    auto color = Piece::Colors(m_moveCounter % 2);
    fprintf(stdout, "%s: Command to %s: '%s%.*s%s'\n", __func__, (color ? "Black" : "White"), "\033[1m", int(command.size()), command.data(), "\033[0m");

    if (command.empty()) return false;
    auto tokens = split(command, ' ');
    auto pos_from = INVALID_POS;
    auto type = Piece::Types::NUM_PIECES;
    if (tokens.size() == 1) {
        type = Piece::Types::Pawn;
        pos_to = strToPos(tokens.front());
    }
    else {
        pos_from = strToPos(tokens.front());
        if (pos_from == INVALID_POS) type = Piece::Types(strToType(tokens.front()));
        pos_to = strToPos(tokens.back());
    }
    if (pos_to == INVALID_POS) return false;
    if (pos_from == INVALID_POS) {
        if (type == Piece::Types::NUM_PIECES) return false;
        auto& pieces = color ? m_state->blacks : m_state->whites;
        for (auto& p : pieces) {
            if (p.type() == type && p.canReach(pos_to)) {
                pos_from = p.pos();
                break;
            }
        }
    }
    if (pos_from == INVALID_POS) return false;
    if (m_state->board[pos_from] == nullptr) return false;
    piece = m_state->board[pos_from];
    if (piece->color() != color) return false;
    return true;
}

void Chessboard::flagUpdates(char pos_from, char pos_to) {
    auto color = Piece::Colors(m_moveCounter % 2);
    auto& enemyPieces = color ? m_state->whites : m_state->blacks;
    auto& ownPieces = color ? m_state->blacks : m_state->whites;
    for (auto& p : enemyPieces) {
        if (p.movePattern(pos_to) || p.movePattern(pos_from)) {
            updatePins(p);
            p.invalidate();
        }
    }

    for (auto& p : ownPieces) {
        if (p.movePattern(pos_to) || p.movePattern(pos_from)) {
            updatePins(p);
            p.invalidate();
        }
    }
}

void Chessboard::updatePins(Piece& piece) {
    if (piece.type() == Piece::Pawn || piece.type() == Piece::Knight || piece.type() == Piece::King) return;
    auto& enemyPieces = piece.color() ? m_state->whites : m_state->blacks;
    auto& enemyPins = piece.color() ? m_state->whitePins : m_state->blackPins;
    auto& king = enemyPieces.k;
    auto it = std::find_if(enemyPins.begin(), enemyPins.end(), [&] (const Pin& pin) { return pin.pinner == &piece; });
    if (it != enemyPins.end()) {
        it->pinned->invalidate();
        enemyPins.erase(it);
    }
    if (piece.movePattern(king.pos())) {
        auto to = positions[king.pos()];
        auto from = piece.coord();
        Direction d = normalize({char(to[R] - from[R]), char(to[F] - from[F])});

        auto reached = traverse(piece.pos(), d, Find(m_state->board));
        auto foundPiece = m_state->board[reached];
        if (&king == foundPiece) {
            // check
            king.invalidate();
        }
        else if (foundPiece && foundPiece->color() != piece.color()) {
            reached = traverse(reached, d, Find(m_state->board));
            if (&king == m_state->board[reached]) {
                enemyPins.push_back({d, &piece, foundPiece});
                foundPiece->invalidate();
            }
        }
    }
}

void Chessboard::detectChecks() {
    auto color = Piece::Colors(m_moveCounter % 2);
    auto& enemyPieces = color ? m_state->whites : m_state->blacks;
    auto& ownPieces = color ? m_state->blacks : m_state->whites;
    auto& king = enemyPieces.k;
    auto& pawnAttackLeft = color ? SW : NW;
    auto& pawnAttackRight = color ? SE : NE;
    for (auto& p : ownPieces) {
        if (!p.movePattern(king.pos())) continue;
        auto to = positions[king.pos()];
        auto from = p.coord();

        if (p.type() == Piece::Knight) {
            if (!m_inCheck) {
                m_allowedInCheck = { p.pos() };
            }
            else {
                m_allowedInCheck.clear();
            }
            m_inCheck = true;
        }
        else if (p.type() == Piece::Pawn) {
            Direction d {char(to[R] - from[R]), char(to[F] - from[F])};
            if (d == pawnAttackLeft || d == pawnAttackRight) {
                if (!m_inCheck) {
                    m_allowedInCheck = { p.pos() };
                }
                else {
                    m_allowedInCheck.clear();
                }
                m_inCheck = true;
            }
        }
        else {
            Direction d = normalize({char(to[R] - from[R]), char(to[F] - from[F])});
            std::set<char> tmp;
            auto pos = traverse(p.pos(), d, Add(m_state->board, tmp, king.color()));
            if (pos == king.pos()) {
                tmp.insert(p.pos());
                if (!m_inCheck) {
                    m_allowedInCheck = std::move(tmp);
                }
                else {
                    m_allowedInCheck.clear();
                }
                m_inCheck = true;
            }
        }
    }
}

bool Chessboard::move(Piece& piece, char pos_to) {
    auto& allowed = piece.allowed();

    if (allowed.count(pos_to) == 0 || (m_inCheck && piece.type() != Piece::King && m_allowedInCheck.count(pos_to) == 0)) return false;
    if (m_state->board[pos_to] && m_state->board[pos_to]->color() == piece.color()) return false;
    if (m_state->board[pos_to]) m_state->board[pos_to]->take();
    m_state->board[piece.pos()] = nullptr;
    m_state->board[pos_to] = &piece;
    piece.setPos(pos_to);

    m_inCheck = false;
    m_allowedInCheck.clear();

    return true;
}
