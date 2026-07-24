// Minimal Qwen2 byte-level BPE tokenizer (GPT-2 style) for the CosyVoice3 text
// frontend. Loads vocab.json + merges.txt (from CosyVoice-BlankEN) and matches
// the HF AutoTokenizer's encode() for the text CosyVoice feeds the LM.
//
// Supports: byte-level encoding, ranked BPE merges, the Qwen2 pre-tokenizer
// split (unicode-letter approximated as ASCII-letter | any non-ASCII), and a
// small set of special tokens (only those that actually appear in prompts,
// e.g. <|endofprompt|>). Not a general-purpose tokenizer.
#pragma once
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

struct QwenTokenizer {
    std::unordered_map<std::string,int> vocab;      // token string -> id
    std::unordered_map<std::string,int> merge_rank; // "A B" -> rank
    std::string byte2u[256];                        // byte -> utf8 of mapped codepoint
    std::vector<std::pair<std::string,int>> specials; // (string,id), longest first

    static std::string utf8_of(uint32_t cp){
        std::string s;
        if(cp<0x80) s+=(char)cp;
        else if(cp<0x800){ s+=(char)(0xC0|(cp>>6)); s+=(char)(0x80|(cp&0x3F)); }
        else if(cp<0x10000){ s+=(char)(0xE0|(cp>>12)); s+=(char)(0x80|((cp>>6)&0x3F)); s+=(char)(0x80|(cp&0x3F)); }
        else { s+=(char)(0xF0|(cp>>18)); s+=(char)(0x80|((cp>>12)&0x3F)); s+=(char)(0x80|((cp>>6)&0x3F)); s+=(char)(0x80|(cp&0x3F)); }
        return s;
    }
    // GPT-2 bytes_to_unicode
    void build_byte_map(){
        std::vector<int> bs, cs;
        for(int b=33;b<=126;++b){ bs.push_back(b); cs.push_back(b); }
        for(int b=161;b<=172;++b){ bs.push_back(b); cs.push_back(b); }
        for(int b=174;b<=255;++b){ bs.push_back(b); cs.push_back(b); }
        int n=0;
        for(int b=0;b<256;++b){ if(std::find(bs.begin(),bs.end(),b)==bs.end()){ bs.push_back(b); cs.push_back(256+n); ++n; } }
        for(size_t i=0;i<bs.size();++i) byte2u[bs[i]] = utf8_of((uint32_t)cs[i]);
    }

    bool load(const std::string& vocab_json, const std::string& merges_txt){
        build_byte_map();
        // vocab.json: {"token": id, ...}. Parse manually (values are ints, keys JSON strings).
        std::ifstream vf(vocab_json); if(!vf) return false;
        std::string all((std::istreambuf_iterator<char>(vf)), {});
        parse_vocab(all);
        // merges.txt: skip first line if "#version", then "A B" per line ranked.
        std::ifstream mf(merges_txt); if(!mf) return false;
        std::string line; int rank=0; bool first=true;
        while(std::getline(mf,line)){
            if(first && line.size()>=1 && line[0]=='#'){ first=false; continue; }
            first=false;
            if(line.empty()) continue;
            size_t sp=line.find(' '); if(sp==std::string::npos) continue;
            merge_rank[line]=rank++;
        }
        // special tokens actually used in CosyVoice prompts
        add_special("<|endofprompt|>", 151646);
        add_special("<|endoftext|>", 151643);
        add_special("<|im_start|>", 151644);
        add_special("<|im_end|>", 151645);
        std::sort(specials.begin(),specials.end(),
                  [](auto&a,auto&b){return a.first.size()>b.first.size();});
        return true;
    }
    void add_special(const std::string& s, int id){ specials.push_back({s,id}); }

    // Minimal JSON object parser for {"tok": int, ...} with \uXXXX / escapes.
    void parse_vocab(const std::string& s){
        size_t i=0; size_t n=s.size();
        auto skipws=[&]{ while(i<n && (s[i]==' '||s[i]=='\n'||s[i]=='\t'||s[i]=='\r')) ++i; };
        skipws(); if(i<n && s[i]=='{') ++i;
        while(i<n){
            skipws(); if(i<n && s[i]=='}') break;
            if(i<n && s[i]==',' ){ ++i; continue; }
            if(i>=n || s[i]!='"') { ++i; continue; }
            std::string key = parse_jstr(s,i);
            skipws(); if(i<n && s[i]==':') ++i; skipws();
            // integer value
            size_t j=i; if(j<n && (s[j]=='-')) ++j; while(j<n && s[j]>='0'&&s[j]<='9') ++j;
            int val = std::atoi(s.substr(i,j-i).c_str()); i=j;
            vocab[key]=val;
        }
    }
    // parse a JSON string starting at s[i]=='"', advancing i past closing quote.
    static std::string parse_jstr(const std::string& s, size_t& i){
        std::string out; ++i; // skip opening "
        while(i<s.size()){
            char c=s[i++];
            if(c=='"') break;
            if(c=='\\'){
                char e=s[i++];
                switch(e){
                    case 'n': out+='\n'; break; case 't': out+='\t'; break;
                    case 'r': out+='\r'; break; case 'b': out+='\b'; break;
                    case 'f': out+='\f'; break; case '/': out+='/'; break;
                    case '\\': out+='\\'; break; case '"': out+='"'; break;
                    case 'u': {
                        auto hex=[&](int k){ int v=0; for(int t=0;t<4;++t){ char h=s[i+k+t]; v=v*16+(h<='9'?h-'0':(h|32)-'a'+10);} return v; };
                        uint32_t cp=hex(0); i+=4;
                        if(cp>=0xD800 && cp<=0xDBFF && i+1<s.size() && s[i]=='\\' && s[i+1]=='u'){
                            i+=2; uint32_t lo=hex(0); i+=4; cp=0x10000+((cp-0xD800)<<10)+(lo-0xDC00);
                        }
                        out+=utf8_of(cp); break;
                    }
                    default: out+=e; break;
                }
            } else out+=c;
        }
        return out;
    }

    // ---- UTF-8 codepoint decode ----
    static std::vector<uint32_t> utf8_decode(const std::string& s){
        std::vector<uint32_t> cps; size_t i=0, n=s.size();
        while(i<n){ unsigned char c=s[i]; uint32_t cp; int len;
            if(c<0x80){cp=c;len=1;} else if((c>>5)==6){cp=c&0x1F;len=2;}
            else if((c>>4)==14){cp=c&0x0F;len=3;} else {cp=c&0x07;len=4;}
            for(int k=1;k<len&&i+k<n;++k) cp=(cp<<6)|(s[i+k]&0x3F);
            cps.push_back(cp); i+=len; }
        return cps;
    }
    static bool is_letter(uint32_t cp){ return (cp>='a'&&cp<='z')||(cp>='A'&&cp<='Z')||cp>=0x80; }
    static bool is_num(uint32_t cp){ return cp>='0'&&cp<='9'; }
    static bool is_space(uint32_t cp){ return cp==' '||cp=='\t'||cp=='\n'||cp=='\r'||cp==0x0b||cp==0x0c; }
    static bool is_nl(uint32_t cp){ return cp=='\r'||cp=='\n'; }

    // Qwen2 pre-tokenizer split of one (special-free) text span -> byte pieces.
    std::vector<std::string> pretokenize(const std::string& text){
        std::vector<uint32_t> cp = utf8_decode(text);
        std::vector<std::string> pieces; size_t i=0, n=cp.size();
        auto emit=[&](size_t a, size_t b){ std::string p; for(size_t k=a;k<b;++k) p+=utf8_of(cp[k]); pieces.push_back(p); };
        while(i<n){
            // 1. contraction 's 't 're 've 'm 'll 'd (case-insensitive)
            if(cp[i]=='\''){
                auto lc=[&](size_t k){ uint32_t c=(k<n)?cp[k]:0; return (c>='A'&&c<='Z')?c+32:c; };
                if(i+1<n){ uint32_t a=lc(i+1);
                    if(a=='s'||a=='t'||a=='m'||a=='d'){ emit(i,i+2); i+=2; continue; }
                    if(i+2<n){ uint32_t b=lc(i+2);
                        if((a=='r'&&b=='e')||(a=='v'&&b=='e')||(a=='l'&&b=='l')){ emit(i,i+3); i+=3; continue; } } }
            }
            // 2. [^\r\n\p{L}\p{N}]? \p{L}+
            if(is_letter(cp[i])){ size_t j=i; while(j<n && is_letter(cp[j])) ++j; emit(i,j); i=j; continue; }
            if(!is_nl(cp[i]) && !is_num(cp[i]) && i+1<n && is_letter(cp[i+1])){
                size_t j=i+1; while(j<n && is_letter(cp[j])) ++j; emit(i,j); i=j; continue; }
            // 3. \p{N} (single digit)
            if(is_num(cp[i])){ emit(i,i+1); ++i; continue; }
            // 4. ' ?[^\s\p{L}\p{N}]+[\r\n]*'
            { size_t j=i; if(cp[j]==' ' && j+1<n && !is_space(cp[j+1]) && !is_letter(cp[j+1]) && !is_num(cp[j+1])) ++j;
              if(j<n && !is_space(cp[j]) && !is_letter(cp[j]) && !is_num(cp[j])){
                  while(j<n && !is_space(cp[j]) && !is_letter(cp[j]) && !is_num(cp[j])) ++j;
                  while(j<n && is_nl(cp[j])) ++j; emit(i,j); i=j; continue; } }
            // 5/6/7. whitespace runs
            if(is_space(cp[i])){ size_t j=i; while(j<n && is_space(cp[j])) ++j;
                // \s+(?!\S): if followed by non-space, leave the last space for the next word
                if(j<n && j-i>1) --j; emit(i,j); i=j; continue; }
            emit(i,i+1); ++i; // fallback
        }
        return pieces;
    }

    std::vector<std::string> bpe(const std::string& piece){
        std::vector<std::string> word;
        for(unsigned char b : piece) word.push_back(byte2u[b]);
        if(word.size()<2) return word;
        while(true){
            int best=1e9; size_t bi=0; bool found=false;
            for(size_t i=0;i+1<word.size();++i){
                auto it=merge_rank.find(word[i]+" "+word[i+1]);
                if(it!=merge_rank.end() && it->second<best){ best=it->second; bi=i; found=true; }
            }
            if(!found) break;
            word[bi]=word[bi]+word[bi+1];
            word.erase(word.begin()+bi+1);
        }
        return word;
    }

    std::vector<int> encode(const std::string& text){
        std::vector<int> ids;
        // split out special tokens first (longest match)
        size_t pos=0;
        while(pos<text.size()){
            size_t best=std::string::npos, blen=0; int bid=0;
            for(auto& sp:specials){ size_t f=text.find(sp.first,pos);
                if(f!=std::string::npos && (best==std::string::npos || f<best)){ best=f; blen=sp.first.size(); bid=sp.second; } }
            size_t end = (best==std::string::npos)? text.size() : best;
            if(end>pos){ std::string span=text.substr(pos,end-pos);
                for(auto& pc: pretokenize(span)) for(auto& sym: bpe(pc)){
                    auto it=vocab.find(sym); if(it!=vocab.end()) ids.push_back(it->second);
                    // Byte-level BPE with a complete vocab makes every symbol
                    // representable, so an OOV here means a bad/mismatched
                    // vocab.json. Fail loudly rather than dropping a token and
                    // silently corrupting the LM input (-> wrong audio).
                    else throw std::runtime_error("qwen_tokenizer: OOV BPE symbol '" + sym + "' (vocab.json incomplete or mismatched)");
                } }
            if(best==std::string::npos) break;
            ids.push_back(bid); pos=best+blen;
        }
        return ids;
    }
};
