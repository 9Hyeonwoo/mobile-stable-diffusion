//
// Created by 구현우 on 2023/11/19.
//

#include "tokenizer.h"
#include <fstream>

#define LOG_TAG "TOKENIZER"

#define MEDIA_PATH(filename) "/sdcard/Android/media/com.example.myopencl/" #filename

SimpleTokenizer::SimpleTokenizer() {
    pat = std::regex(
            R"(<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[\w]+|[\d]|[^\s\w\d]+)",
            std::regex::icase);
    // below is the original for wstring.
    // pat = std::wregex(L"<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+", std::wregex::icase);
    cache.insert(std::make_pair("<start_of_text>", "<start_of_text>"));
    cache.insert(std::make_pair("<end_of_text>", "<end_of_text>"));

    auto byteToUnicode = bytes_to_unicode();
    for (auto pair: byteToUnicode) {
        byte_encoder.insert(pair);
    }

    std::vector<std::string> vocab;
    for (auto &pair: byteToUnicode) {
        vocab.push_back(pair.second);
    }
    for (auto &pair: byteToUnicode) {
        vocab.push_back(pair.second + "</w>");
    }

    // read vocab file
    std::ifstream file(MEDIA_PATH(bpe_simple_vocab_16e6.txt));
    if (!file.is_open()) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "Failed to open the %s", MEDIA_PATH("bpe_simple_vocab_16e6.txt"));
        throw std::runtime_error("Failed to open the file.");
    }

    std::string line;
    int i = 0;
    while (std::getline(file, line)) {
        if (i == 0) {
            i++;
            continue;
        }
        if (i == 49152 - 256 - 2 + 1) {
            break;
        }

        std::istringstream ss2(line);
        std::string first;
        std::string second;
        std::getline(ss2, first, ' ');
        std::getline(ss2, second, ' ');

        vocab.push_back(first + second);
        bpe_ranks.insert(std::make_pair(std::make_pair(first, second), i - 1));
        i++;
    }
    file.close();

    vocab.emplace_back("<start_of_text>");
    vocab.emplace_back("<end_of_text>");

    for (i = 0; i < vocab.size(); i++) {
        encoder.insert(std::make_pair(vocab[i], i));
    }
}

SimpleTokenizer::~SimpleTokenizer() = default;

/*
 * @param text: only alphabet and dot/comma/etc characters with stripped.
 *              (expect basic_clean&whitespace_clean applied)
 */
std::vector<int> SimpleTokenizer::encode(std::string text) {
    std::vector<int> bpe_tokens;
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

    // `text` to lower case.
    std::transform(text.begin(), text.end(), text.begin(), ::tolower);

    std::smatch match;
    while (std::regex_search(text, match, pat)) {
        // Expect match.str() is UTF-8 encoding.
        std::string token;

        // __android_log_print(ANDROID_LOG_DEBUG, "__TEST__", "match: %s", match.str().c_str());

        // transform token to byte encoding.
        for (unsigned char b: match.str()) {
            token += byte_encoder[static_cast<int>(b)];
        }

        // collect bpe tokens.
        std::string bpe_token = bpe(token);
        std::istringstream ss(bpe_token);
        std::string bpe_split;

        while (std::getline(ss, bpe_split, ' ')) {
            bpe_tokens.push_back(encoder[bpe_split]);
        }

        text = match.suffix();
    }
    return bpe_tokens;
}

std::string SimpleTokenizer::bpe(std::string token) {
    if (cache.find(token) != cache.end()) {
        return cache[token];
    }

    std::vector<std::string> word;
    for (int i = 0; i < token.size() - 1; i++) {
        word.emplace_back(1, token[i]);
    }
    auto tmp = std::string(1, token[token.size() - 1]) + "</w>";
    word.push_back(tmp);

    // get pairs of adjacent characters.
    std::vector<std::pair<std::string, std::string>> pairs = get_pairs(word);

    if (pairs.empty()) {
        return token + "</w>";
    }

    while (true) {
        auto bigram = *std::min_element(pairs.begin(), pairs.end(), [this](auto a, auto b) {
            auto aValue = (bpe_ranks.find(a) != bpe_ranks.end()) ? bpe_ranks[a]
                                                                 : std::numeric_limits<int>::max();
            auto bValue = (bpe_ranks.find(b) != bpe_ranks.end()) ? bpe_ranks[b]
                                                                 : std::numeric_limits<int>::max();
            return aValue < bValue;
        });

        if (bpe_ranks.find(bigram) == bpe_ranks.end()) {
            break;
        }

        std::vector<std::string> new_word;
        for (int i = 0; i < word.size();) {
            auto j = std::find(word.begin() + i, word.end(), bigram.first);
            if (j != word.end()) {
                for (auto w = word.begin() + i; w != j; w++) {
                    new_word.push_back(*w);
                }
                i = j - word.begin();
            } else {
                for (auto w = word.begin() + i; w != word.end(); w++) {
                    new_word.push_back(*w);
                }
                break;
            }

            if (word[i] == bigram.first && i < word.size() - 1 && word[i + 1] == bigram.second) {
                new_word.push_back(bigram.first + bigram.second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }

        word = new_word;
        if (word.size() == 1) {
            break;
        } else {
            pairs = get_pairs(word);
        }
    }

    auto joined_word = std::accumulate(
            word.begin(), word.end(), std::string(),
            [](const std::string &a, const std::string &b) {
                return a.empty() ? b : a + " " + b;
            }
    );

    cache[token] = joined_word;

    return joined_word;
}

std::vector<std::pair<std::string, std::string>>
SimpleTokenizer::get_pairs(std::vector<std::string> word) {
    std::vector<std::pair<std::string, std::string>> pairs;
    for (int i = 1; i < word.size(); i++) {
        pairs.emplace_back(word[i - 1], word[i]);
    }
    return pairs;
}

std::vector<std::pair<int, std::string>> SimpleTokenizer::bytes_to_unicode() {
    std::vector<std::pair<int, std::string>> byteToUnicode;
    std::wstring wstr;
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

    // Range from '!' to '~'
    for (int b = static_cast<int>(L'!'); b <= static_cast<int>(L'~'); ++b) {
        wstr = b;
        byteToUnicode.emplace_back(b, converter.to_bytes(wstr));
    }

    // Range from '¡' to '¬'
    for (int b = static_cast<int>(L'¡'); b <= static_cast<int>(L'¬'); ++b) {
        wstr = b;
        byteToUnicode.emplace_back(b, converter.to_bytes(wstr));
    }

    // Range from '®' to 'ÿ'
    for (int b = static_cast<int>(L'®'); b <= static_cast<int>(L'ÿ'); ++b) {
        wstr = b;
        byteToUnicode.emplace_back(b, converter.to_bytes(wstr));
    }

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find_if(byteToUnicode.begin(), byteToUnicode.end(),
                         [b](const auto &pair) { return pair.first == b; }) ==
            byteToUnicode.end()) {
            wstr = 256 + n;
            byteToUnicode.emplace_back(b, converter.to_bytes(wstr));
            ++n;
        }
    }

    return byteToUnicode;
}

std::vector<long> SimpleTokenizer::tokenize(const std::string &text) {
    return tokenize(std::vector<std::string>{text});
}

std::vector<long> SimpleTokenizer::tokenize(const std::vector<std::string> &texts) {
    std::vector<long> result;
    const auto sot_token = encoder["<start_of_text>"];
    const auto eot_token = encoder["<end_of_text>"];

    for (auto &text: texts) {
        auto tokens = encode(text);
        tokens.insert(tokens.begin(), sot_token);
        tokens.push_back(eot_token);
        if (tokens.size() > CONTEXT_LENGTH) {
            tokens[CONTEXT_LENGTH - 1] = eot_token;
        }
        tokens.resize(CONTEXT_LENGTH, 0);

        result.insert(result.end(), tokens.begin(), tokens.end());
    }

    return result;
}