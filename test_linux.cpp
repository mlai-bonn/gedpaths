// filepath: /home/florian/Documents/CodeProjectsGit/GNNGED/test_linux.cpp
// Simple check/test for the Precomputed LINUX mappings/graphs files.

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <regex>

int main(int argc, char** argv) {
    namespace fs = std::filesystem;

    // Allow optional path override via first argument. If not provided,
    // attempt to locate the precomputed folder by searching upwards from
    // the current working directory (and from the executable path).
    fs::path candidate;
    if (argc > 1) {
        candidate = fs::path(argv[1]);
    } else {
        // search up to a few parent directories for the expected relative path
        fs::path cwd = fs::current_path();
        std::vector<fs::path> start_points = {cwd};
        // also try the executable directory if available
        try {
            fs::path exe = fs::read_symlink("/proc/self/exe");
            start_points.push_back(exe.parent_path());
        } catch (...) {
            // ignore if not available
        }

        bool found = false;
        for (const auto &sp : start_points) {
            fs::path p = sp;
            for (int i = 0; i < 6; ++i) {
                fs::path cand = p / "Results" / "Mappings" / "Precomputed" / "LINUX";
                if (fs::exists(cand) && fs::is_directory(cand)) {
                    candidate = cand;
                    found = true;
                    break;
                }
                if (p.has_parent_path()) p = p.parent_path();
                else break;
            }
            if (found) break;
        }
        if (!found) {
            // fallback to the simple relative path
            candidate = fs::path("Results/Mappings/Precomputed/LINUX");
        }
    }

    fs::path base = candidate;
    std::cout << "Checking precomputed LINUX folder: " << base << std::endl;
    if (!fs::exists(base) || !fs::is_directory(base)) {
        std::cerr << "Error: directory not found: " << base << std::endl;
        std::cerr << "Try running: ./build/TestLINUX /absolute/path/to/Results/Mappings/Precomputed/LINUX" << std::endl;
        return 1;
    }

    fs::path matchings = base / "matchings.txt";
    fs::path graphs = base / "graphs.txt";
    if (!fs::exists(matchings)) {
        std::cerr << "Error: matchings.txt not found in " << base << std::endl;
        return 2;
    }
    if (!fs::exists(graphs)) {
        std::cerr << "Warning: graphs.txt not found in " << base << " (this may be optional)" << std::endl;
    }

    std::ifstream ifs(matchings);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open " << matchings << std::endl;
        return 3;
    }

    std::string line;
    size_t header_count = 0;
    std::regex header_re(R"(^\s*(\d+)\s+(\d+)\s+(\d+)\b)");
    std::smatch m;

    // We'll also validate the first block in detail
    bool validated_first_block = false;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        if (line[0] == '#') continue;
        if (std::regex_search(line, m, header_re)) {
            ++header_count;
            if (!validated_first_block) {
                // parse header
                int source = std::stoi(m[1].str());
                int target = std::stoi(m[2].str());
                int num_nodes = std::stoi(m[3].str());
                std::cout << "Found first header: source=" << source << " target=" << target << " num_nodes=" << num_nodes << std::endl;
                // Read next num_nodes non-empty lines and validate they contain integer pairs
                size_t read = 0;
                while (read < static_cast<size_t>(num_nodes) && std::getline(ifs, line)) {
                    if (line.empty()) continue;
                    std::istringstream iss(line);
                    int a, b;
                    if (!(iss >> a >> b)) {
                        std::cerr << "Invalid mapping line for block header at mapping index " << read << ": '" << line << "'\n";
                        return 4;
                    }
                    ++read;
                }
                if (read != static_cast<size_t>(num_nodes)) {
                    std::cerr << "Warning: expected " << num_nodes << " mapping lines but read " << read << std::endl;
                    // not fatal
                }
                validated_first_block = true;
            }
        }
    }

    if (header_count == 0) {
        std::cerr << "No header blocks found in matchings.txt" << std::endl;
        return 5;
    }

    std::cout << "matchings.txt looks good -- found " << header_count << " mapping headers." << std::endl;
    return 0;
}
