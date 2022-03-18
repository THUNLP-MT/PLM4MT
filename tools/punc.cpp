#include <string>
#include <iostream>

void replace_punc(const std::string& str, std::string& out)
{
    int n = str.size();
    int status = 0;

    // Postprocess punctuations
    for (int i = 0; i < n; i++) {
        char c = str[i];

        if ((unsigned char) out.back() < 0x80) {
	          if (i != n - 1 && (unsigned char) str[i + 1] < 0x80) {
	              out += c;
	              continue;
	          }
        }

        switch (c) {
        case '!':
            out += "\xef\xbc\x81";
            break;
        case '(':
            out += "\xef\xbc\x88";
            break;
        case '"':
            out += (status == 0) ? "\xe2\x80\x9c" : "\xe2\x80\x9d";
            status = (status == 0) ? 1 : 0;
            break;
        case ')':
            out += "\xef\xbc\x89";
            break;
        case ',':
            out += "\xef\xbc\x8c";
            break;
        case '.':
            out += "\xe3\x80\x82";
            break;
        case ':':
            out += "\xef\xbc\x9a";
            break;
        case ';':
            out += "\xef\xbc\x9b";
            break;
        case '?':
            out += "\xef\xbc\x9f";
            break;
        case '[':
            out += "\xef\xbc\xbb";
            break;
        case ']':
            out += "\xef\xbc\xbd";
            break;
        case '~':
            out += "\xef\xbd\x9e";
            break;
        default:
            out += c;
            break;
        }
    }
}


int main()
{
    std::string line;

    while (std::getline(std::cin, line)) {
        std::string output;

        replace_punc(line, output);
        std::cout << output << std::endl;
    }
}
