// Table-driven checks of the number normalizers used on Parler prompts
// before tokenization (English words + Indic script-native digits).

#include "parler/text_norm.h"

#include <cstdio>
#include <string>

using tts_cpp::parler::detail::normalize_numbers_en;
using tts_cpp::parler::detail::normalize_numbers_indic;

static int g_failures = 0;

static void expect(const std::string & in, const std::string & want) {
    const std::string got = normalize_numbers_en(in);
    if (got != want) {
        fprintf(stderr, "FAIL: \"%s\"\n  got:  \"%s\"\n  want: \"%s\"\n",
                in.c_str(), got.c_str(), want.c_str());
        ++g_failures;
    }
}

static void expect_indic(const std::string & in, const std::string & want) {
    const std::string got = normalize_numbers_indic(in);
    if (got != want) {
        fprintf(stderr, "FAIL(indic): \"%s\"\n  got:  \"%s\"\n  want: \"%s\"\n",
                in.c_str(), got.c_str(), want.c_str());
        ++g_failures;
    }
}

int main() {
    // digit-free text is byte-identical
    expect("Hey, how are you doing today?", "Hey, how are you doing today?");
    expect("", "");
    expect("no digits at all!", "no digits at all!");

    // cardinals
    expect("12", "twelve");
    expect("0", "zero");
    expect("10", "ten");
    expect("19", "nineteen");
    expect("20", "twenty");
    expect("42", "forty-two");
    expect("100", "one hundred");
    expect("105", "one hundred five");
    expect("999", "nine hundred ninety-nine");
    expect("1000000", "one million");
    expect("1001", "one thousand one");
    expect("999999999999999",
           "nine hundred ninety-nine trillion nine hundred ninety-nine billion "
           "nine hundred ninety-nine million nine hundred ninety-nine thousand "
           "nine hundred ninety-nine");

    // the fixture case1 sentence that motivated the normalizer
    expect("The quick brown fox jumps over 12 lazy dogs, doesn't it?",
           "The quick brown fox jumps over twelve lazy dogs, doesn't it?");

    // thousands separators (and near-misses stay untouched as separators)
    expect("1,234", "one thousand two hundred thirty-four");
    expect("12,345,678",
           "twelve million three hundred forty-five thousand six hundred seventy-eight");
    expect("1,23", "one,twenty-three");
    expect("1234,567", "one thousand two hundred thirty-four,five hundred sixty-seven");
    expect("1,2345", "one,two thousand three hundred forty-five");

    // decimals
    expect("3.14", "three point one four");
    expect("0.5", "zero point five");
    expect("1,234.56", "one thousand two hundred thirty-four point five six");
    expect("12.", "twelve.");

    // ordinals
    expect("1st", "first");
    expect("2nd", "second");
    expect("3rd", "third");
    expect("4th", "fourth");
    expect("5th", "fifth");
    expect("8th", "eighth");
    expect("9th", "ninth");
    expect("12th", "twelfth");
    expect("20th", "twentieth");
    expect("21st", "twenty-first");
    expect("100th", "one hundredth");
    expect("1,000th", "one thousandth");
    expect("4ths", "fourths");
    expect("2/3rds", "two/thirds");
    expect("21st-century", "twenty-first-century");
    expect("1sting", "one sting"); // suffix glued to letters is not an ordinal

    // leading zeros and over-long runs read digit-by-digit
    expect("007", "zero zero seven");
    expect("1234567890123456", // 16 digits
           "one two three four five six seven eight nine zero one two three four five six");

    // spacing against abutting text
    expect("A4", "A four");
    expect("(12)", "(twelve)");
    expect("12:30", "twelve:thirty");
    expect("x=12", "x=twelve");
    expect("caf\xc3\xa9 12", "caf\xc3\xa9 twelve");
    expect("caf\xc3\xa9""12", "caf\xc3\xa9 twelve");

    // idempotence: output has no digits, a second pass changes nothing
    {
        const std::string once  = normalize_numbers_en("Call 12 at 3.14, 21st!");
        const std::string twice = normalize_numbers_en(once);
        if (once != twice) {
            fprintf(stderr, "FAIL: not idempotent: \"%s\" -> \"%s\"\n",
                    once.c_str(), twice.c_str());
            ++g_failures;
        }
    }

    // ---- indic variant: script-native digit transliteration ----
    // Hindi / Devanagari context
    expect_indic("कमरा 12 में", "कमरा १२ में");
    expect_indic("आज १२ तारीख़ है और 12 बज रहे हैं।",
                 "आज १२ तारीख़ है और १२ बज रहे हैं।");
    // Gujarati / Tamil / Urdu contexts
    expect_indic("કુલ 25 રૂપિયા", "કુલ ૨૫ રૂપિયા");
    expect_indic("மொத்தம் 25", "மொத்தம் ௨௫");
    expect_indic("کل 25 روپے", "کل ۲۵ روپے");
    // context from a preceding NATIVE digit (digits live in script blocks)
    expect_indic("१२, 34", "१२, ३४");
    // right-context rescue when nothing scripted precedes
    expect_indic("12 बजे", "१२ बजे");
    // danda is SHARED punctuation (encoded in the Devanagari block) — it must
    // not resolve script context; the Bengali letter beyond it wins
    expect_indic("আজ সোমবার। 12 তারিখ", "আজ সোমবার। ১২ তারিখ");
    expect_indic("ਕੁੱਲ। 45", "ਕੁੱਲ। ੪੫");
    // multiplication sign is not a Latin letter; the Devanagari context wins
    expect_indic("5×3 बजे", "५×३ बजे");
    // malformed UTF-8 bytes pass through and carry no context
    expect_indic("\xE0 12 \xE0\xA4\xAC\xE0\xA4\x9C\xE0\xA5\x87",
                 "\xE0 १२ \xE0\xA4\xAC\xE0\xA4\x9C\xE0\xA5\x87");
    // Latin context falls through to English words; per-run resolution
    expect_indic("Meeting at 12 tomorrow", "Meeting at twelve tomorrow");
    expect_indic("English 12 और हिंदी में 34",
                 "English twelve और हिंदी में ३४");
    // standalone digits have no script context -> English words
    expect_indic("12", "twelve");
    // decimals and separators keep punctuation, digits go native
    expect_indic("कीमत 3.14 है", "कीमत ३.१४ है");
    expect_indic("कुल 1,234 लोग", "कुल १,२३४ लोग");
    // digit-free text is byte-identical
    expect_indic("મારું નામ પ્રતિક છે. કેમ છો?", "મારું નામ પ્રતિક છે. કેમ છો?");
    expect_indic("", "");
    // idempotence: no ASCII digits remain after the pass
    {
        const std::string once  = normalize_numbers_indic("कमरा 12, room 34");
        const std::string twice = normalize_numbers_indic(once);
        if (once != twice) {
            fprintf(stderr, "FAIL(indic): not idempotent: \"%s\" -> \"%s\"\n",
                    once.c_str(), twice.c_str());
            ++g_failures;
        }
    }

    if (g_failures) {
        fprintf(stderr, "test_parler_text_norm: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("test_parler_text_norm: OK\n");
    return 0;
}
