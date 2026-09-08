// Implements a parser for an extended Backus-Naur form (BNF), producing the
// binary context-free grammar format specified by whisper.h. Supports character
// ranges, grouping, and repetition operators. As an example, a grammar for
// arithmetic might look like:
//
// root  ::= expr
// expr  ::= term ([-+*/] term)*
// term  ::= num | "(" space expr ")" space
// num   ::= [0-9]+ space
// space ::= [ \t\n]*

#pragma once
#include "whisper.h"
#include <vector>
#include <map>
#include <cstdint>
#include <string>

namespace grammar_parser {
    struct parse_state {
        std::map<std::string, uint32_t>                   symbol_ids;
        std::vector<std::vector<whisper_grammar_element>> rules;

        std::vector<const whisper_grammar_element *>      c_rules() const;
    };

    parse_state parse(const char * src);
    void print_grammar(FILE * file, const parse_state & state);
}
