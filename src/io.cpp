#include "io.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>

#define LEFTSTRIP 0
#define RIGHTSTRIP 1
#define BOTHSTRIP 2
#define PRECISION 16

std::string do_strip(const std::string &str, int striptype, const std::string &chars) {
    int len = (int) str.size(), i, j, charslen = (int) chars.size();
    if (charslen == 0) {
        i = 0;
        if (striptype != RIGHTSTRIP) {
            while (i < len && ::isspace(str[i])) {
                i++;
            }
        }

        j = len;
        if (striptype != LEFTSTRIP) {
            do {
                j--;
            } while (j >= i && ::isspace(str[j]));
            j++;
        }
    } else {
        const char *sep = chars.c_str();
        i = 0;
        if (striptype != RIGHTSTRIP) {
            while (i < len && memchr(sep, str[i], charslen)) {
                i++;
            }
        }

        j = len;
        if (striptype != LEFTSTRIP) {
            do {
                j--;
            } while (j >= i && memchr(sep, str[j], charslen));
            j++;
        }
    }

    if (i == 0 && j == len) {
        return str;
    } else {
        return str.substr(i, j - i);
    }
}

std::string strip(const std::string &str, const std::string &chars) {
    return do_strip(str, BOTHSTRIP, chars);
}

std::string lstrip(const std::string &str, const std::string &chars) {
    return do_strip(str, LEFTSTRIP, chars);
}

std::string rstrip(const std::string &str, const std::string &chars) {
    return do_strip(str, RIGHTSTRIP, chars);
}

void split_whitespace(const std::string &str, std::vector<std::string> &result, int maxsplit) {
    std::string::size_type i, j, len = str.size();
    for (i = j = 0; i < len;) {
        while (i < len && ::isspace(str[i])) i++;
        j = i;

        while (i < len && !::isspace(str[i])) i++;

        if (j < i) {
            if (maxsplit-- <= 0) break;
            result.push_back(str.substr(j, i - j));
            while (i < len && ::isspace(str[i])) i++;
            j = i;
        }
    }
    if (j < len) {
        result.push_back(str.substr(j, len - j));
    }
}

void split(const std::string &str, std::vector<std::string> &result, const std::string &sep, int maxsplit) {
    result.clear();
    if (maxsplit < 0) maxsplit = INT_MAX;//result.max_size();
    if (sep.size() == 0) {
        split_whitespace(str, result, maxsplit);
        return;
    }

    std::string::size_type i, j, len = str.size(), n = sep.size();
    i = j = 0;
    while (i + n <= len) {
        if (str[i] == sep[0] && str.substr(i, n) == sep) {
            if (maxsplit-- <= 0) break;
            result.push_back(str.substr(j, i - j));
            i = j = i + n;
        } else {
            i++;
        }
    }
    result.push_back(str.substr(j, len - j));
}

std::string zfill(const std::string &str, int width) {
    int len = (int) str.size();
    if (len >= width) {
        return str;
    }
    std::string s(str);
    int fill = width - len;
    s = std::string(fill, '0') + s;

    if (s[fill] == '+' || s[fill] == '-') {
        s[0] = s[fill];
        s[fill] = '0';
    }

    return s;
}

//////////////////////////////////////////////////////////////////////////////////////////////

void DumpTrees(vector<Tree> &trees, const char *path) {
    ofstream outfile;
    outfile.open(path);
    size_t t = 0;
    for (auto &tree : trees) {
        outfile << "Booster[" << t << "]:\n";
        for (auto it : tree.nonleaf) {
            auto v = it.second;
            outfile << "\t" << it.first << "," << v.parent << "," << v.left << "," << v.right << ","
                    << v.column << ",";
            outfile << scientific << setprecision(PRECISION) << v.threshold << endl;
        }

        for (auto &it : tree.leaf) {
            auto v = it.second;
            outfile << "\t\t" << it.first << ",";
            for (int i = 0; i < v.values.size(); i++) {
                if (i < v.values.size() - 1) {
                    outfile << scientific << setprecision(PRECISION) << v.values[i] << ",";
                } else {
                    outfile << scientific << setprecision(PRECISION) << v.values[i] << endl;
                }
            }
        }
        ++t;
    }
    outfile.close();
}

void LoadTrees(vector<Tree> &trees, const char *path) {
    ifstream infile(path);
    string line;
    vector<string> contents;
    Tree tree_(false);
    int t = 0, num;
    while (getline(infile, line)) {
        //Booster
        if (line.find("Booster") == 0) {
            if (t > 0) {
                trees.push_back(tree_);
                tree_.clear();
            }
            ++t;
        } else {
            contents.resize(0);
            line = lstrip(line, "\t");
            split(line, contents, ",");
            num = stoi(contents[0]);
            if (num < 0) {
                //nonleaf
                NonLeafNode node;
                node.parent = stoi(contents[1]);
                node.left = stoi(contents[2]);
                node.right = stoi(contents[3]);
                node.column = stoi(contents[4]);
                node.threshold = stod(contents[5]);
                tree_.nonleaf.emplace(num, node);
            } else {
                //leaf
                LeafNode node;
                node.values.resize(contents.size() - 1);
                for (int i = 1; i < contents.size(); ++i) { node.values[i - 1] = stod(contents[i]); }
                tree_.leaf.emplace(num, node);
            }
        }
    }
    infile.close();
}