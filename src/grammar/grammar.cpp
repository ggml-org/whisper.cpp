#include "framework.h"
#include "grammar.h"
#include "grammar-parser.h"

static grammar_parser::whisper_grammar_element** MallocGrammar(grammar_parser::whisper_grammar_element** rules, int n_rules)
{
    // Get total number of whisper_grammar_element.
    const grammar_parser::whisper_grammar_element* posIn;
    auto n_types_total = 0;
    for (size_t i = 0; i < n_rules; i++)
    {
        int n_types = 0;
        for (posIn = rules[i]; ; posIn++)
        {
            n_types++;

            if (posIn->type == grammar_parser::whisper_gretype::WHISPER_GRETYPE_END)
                break;
        }
        n_types_total += n_types;
    }

    // Malloc a single block of memory to contain pointer array and all whisper_grammar_element entries.
    int sizePointers = n_rules * sizeof(void*);
    int sizeData = n_types_total * sizeof(grammar_parser::whisper_grammar_element);
    auto dataCopy = (grammar_parser::whisper_grammar_element**)malloc(sizePointers + sizeData);

    // Copy to allocated memory.
    auto p = (const grammar_parser::whisper_grammar_element**)dataCopy;
    auto posOut = (grammar_parser::whisper_grammar_element*)((size_t)dataCopy + sizePointers);
    for (int i = 0; i < n_rules; i++)
    {
        p[i] = posOut;

        for (posIn = rules[i]; ; posIn++, posOut++)
        {
            *posOut = *posIn;

            if (posIn->type == grammar_parser::whisper_gretype::WHISPER_GRETYPE_END)
                break;
        }
    }

    return dataCopy;
}

//
// Whisper.cpp input is a whisper_grammar_element*[].
// Allocate contiguous memory with whisper_grammar_element*[] followed by each rule's whisper_grammar_element[].
// This provides a convenient way to allocate a grammar and later free() it. Whisper.Net and other language bindings can use this to easily manage grammar allocations.
//
// An example .net binding:
//  extern "C" __declspec(dllexport) void GetGrammar(const char* src, const char* topLevelRule, void** _grammar_rules, int* _n_grammar_rules, int* _i_start_rule);
//
void GetGrammar(const char* src, const char* topLevelRule, void** _grammar_rules, int* _n_grammar_rules, int* _i_start_rule)
{
    auto grammar_parsed = grammar_parser::parse(src);
    auto grammar_rules = grammar_parsed.c_rules();

    auto rules = (grammar_parser::whisper_grammar_element**)grammar_rules.data();
    int n_rules = grammar_rules.size();

    *_grammar_rules = MallocGrammar(rules, n_rules);
    *_n_grammar_rules = n_rules;
    *_i_start_rule = grammar_parsed.symbol_ids.at(topLevelRule);
}

